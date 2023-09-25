import sys
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Flatten, Dense, MaxPooling3D, LSTM
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import scipy.signal
import time
import argparse, gym_unrealcv
import os
from datetime import datetime

#the get_batches function is a generator function that splits the given dataset into smaller batches of a specific size.
def get_batches(dataset, batch_size):
    """Yield successive batches from the dataset."""
    for i in range(0, dataset.shape[0], batch_size):
        #yield keyword is used in Python to define a generator function that can be paused and resumed, allowing it to generate a sequence of results over time, rather than computing them all at once and returning them in a list for instance. 
        yield dataset[i:i + batch_size]

#preprocess function converts it to grayscale, and make its value between -1 and 1 (normalization)
def preprocess(observation):
    # Add an extra dimension for batch size
    image = tf.expand_dims(observation, axis=0)

    # Use the VGG16 preprocess_input method
    image = preprocess_input(image)

    # Remove the batch size dimension anymore since we will immediately pass this to CNN
    image = tf.squeeze(image, axis=0)
    return image

def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        #the * operator can be used to "unpack" the list or tuple into separate arguments.
        #here, frame.shape returns a tuple representing the shape of the frame (for example, (height, width)). So, *frame.shape would unpack this tuple into separate arguments.
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx,:] = frame
    else:
        stacked_frames[0:buffer_size-1,:] = stacked_frames[1:,:]
        stacked_frames[buffer_size-1, :] = frame

    return stacked_frames    

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, *observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        #"cleaning" buffer here
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )
#end of the class buffer

def create_cnn(observation_dimensions):
    # Input is a 4D tensor with shape (frames, height, width, channels) and batch size is always 1
    input_tensor = keras.Input(shape=observation_dimensions)
    
    # Load VGG16 model
    vgg16_model = VGG16(input_shape=observation_dimensions[1:], weights='imagenet', include_top=False)
    
    # We set the layers of VGG16 to be not trainable
    for layer in vgg16_model.layers:
        layer.trainable = False
    
    # We use the VGG16 model in a TimeDistributed manner
    x = keras.layers.TimeDistributed(vgg16_model)(input_tensor)
    
    # Flatten the output of the TimeDistributed VGG16 model
    x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)
    print(x.shape)

    model = keras.models.Model(inputs=input_tensor, outputs=x)
    
    return model


def create_actor_critic(feature_dimensions, num_actions):

    # Input is a 2D tensor with shape (frames, 49 * 512)
    input_tensor = keras.Input(shape=feature_dimensions)
    
    # not add a Dense layer to compact the output dimension since, in seed paper, they don't have this step
    # what is more, In the context of processing video frames for reinforcement learning, a common approach is to use Conv3D or a combination of TimeDistributed with Conv2D for spatial feature extraction and then an LSTM or GRU layer for temporal dynamics learning.
    # A Dense layer is usually not used in between because it would flatten the spatial structure before the temporal dynamics are learned.
    # Flatten and reshape for LSTM layer -- shape: (batchSize, flattened)
    
    # LSTM layer
    x = LSTM(units=512, return_sequences=False)(input_tensor)

    # Output layer
    x = Dense(num_actions, activation=None)(x)  ##note! we just need raw value; we don't need to softmax for actor at this. Critic will generate real values not percentage

    model = keras.models.Model(inputs=input_tensor, outputs=x)
    
    return model

    
def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    features_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        actor_logits = actor(features_buffer)
        
        ratio = tf.exp(
            logprobabilities(actor_logits, action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    new_actor_logits = actor(features_buffer)

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(new_actor_logits, action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(features_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        critic_logits = critic(features_buffer)
        value_loss = tf.reduce_mean((return_buffer - critic_logits) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

# Hyperparameters of the PPO algorithm
steps_per_epoch = 100 #store memories of 100 steps
#batch_size = 1
epochs = 30000
gamma = 0.9
clip_ratio = 0.2
policy_learning_rate = 1e-4
value_function_learning_rate = 1e-4
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01

# True if you want to render the environment
render = False
#the frist agent is the tracker
agentIndex = 0
'''may continue training by making it ture'''
load_checkpoint = False
#inpSize = 336
inpSize = 224
stack_size = 3
channelNum = 3
parser = argparse.ArgumentParser(description=None)
parser.add_argument("-e", "--env_id", nargs='?', default='UnrealSearch-RealisticRoomDoor-DiscreteColor-v0', #'UnrealArm-DiscreteRgbd-v0', #'RobotArm-Discrete-v0',
                    help='Select the environment to run')
parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
args = parser.parse_args()
env = gym.make(args.env_id, resolution=(inpSize, inpSize))
print("Env created")
### should i set here or, as the train.py code, set in the beginning of each episode before env.reset??
env.seed(0)

num_actions = env.action_space[agentIndex].n
# Initialize the buffer

observation_dimensions = (stack_size, inpSize, inpSize, channelNum)
cnnFeatures_dimensions = (stack_size, 49 * 512)
buffer = Buffer(cnnFeatures_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
cnnDecoder = create_cnn(observation_dimensions)
actor = create_actor_critic(cnnFeatures_dimensions, num_actions)
critic = create_actor_critic(cnnFeatures_dimensions, 1)

load_dir_path = 'PPOVggMem100/checkpoints'
dir_path = 'PPOVggMem100/checkpoints'
f = open('PPOVggMem100.txt', 'a')
# Check if the directory exists
if not os.path.exists(load_dir_path):
    # If the directory does not exist, create it
    print("!!!Successfully created the load directory")
    os.makedirs(load_dir_path)
if not os.path.exists(dir_path):
    # If the directory does not exist, create it
    print("!!!Successfully created the save directory")
    os.makedirs(dir_path)

load_actor_checkpoint_file = os.path.join(load_dir_path, 'actor.ckpt')
load_critic_checkpoint_file = os.path.join(load_dir_path, 'critic.ckpt')

actor_checkpoint_file = os.path.join(dir_path, 'actor.ckpt')
critic_checkpoint_file = os.path.join(dir_path, 'critic.ckpt')

if load_checkpoint:
    print('...Loading Checkpoint...')
    actor.load_weights(load_actor_checkpoint_file)
    critic.load_weights(load_critic_checkpoint_file)

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)


'''Begin to train'''
# Initialize the observation, episode return and episode length
observations, episode_return, episode_length = env.reset(), 0, 0
curr_observation = preprocess(observations[agentIndex])
stacked_frames = None
stacked_frames = stack_frames(stacked_frames, curr_observation, stack_size)
# ------ train ------
# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        if render:
            env.render()

        # reshape to have a batch dimension with size of 1 in order to be fed into neural network
        reshaped_stacked_frames = stacked_frames.reshape(1, *stacked_frames.shape)
        reshaped_stacked_features = cnnDecoder(reshaped_stacked_frames) #reshape_... has batch dimension
        # Get the logits, action, and take one step in the environment
        logits, action = sample_action(reshaped_stacked_features)
        # action[0] -- the action of first batch
        observations, rewards, done, _ = env.step(action[0].numpy())
        reward = rewards[agentIndex]
        episode_return += reward
        episode_length += 1
        
        # Get the value and log-probability of the action
        value_t = critic(reshaped_stacked_features)
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        stacked_features = tf.squeeze(reshaped_stacked_features, axis=0)
        #print(stacked_features.shape)
        buffer.store(stacked_features, action, reward, value_t, logprobability_t)

        # Update the observation
        curr_observation = preprocess(observations[agentIndex])
        stacked_frames = stack_frames(stacked_frames, curr_observation, stack_size)

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            reshaped_stacked_frames = stacked_frames.reshape(1, *stacked_frames.shape)
            reshaped_stacked_features = cnnDecoder(reshaped_stacked_frames)
            last_value = 0 if done else critic(reshaped_stacked_features)
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observations, episode_return, episode_length = env.reset(), 0, 0
            curr_observation = preprocess(observations[agentIndex])
            stacked_frames = None
            stacked_frames = stack_frames(stacked_frames, curr_observation, stack_size)
        
    print(f"Epoch: {epoch + 1}.")
    # Get values from the buffer
    (
        featrues_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()
   
    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(
            featrues_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break
    
    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(featrues_buffer, return_buffer)

    featrues_buffer = None
    action_buffer = None
    advantage_buffer = None
    return_buffer = None
    logprobability_buffer = None

    # Print mean return and length for each epoch
    current_time = datetime.now().isoformat()
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}. Time: {current_time}"
    )
    
    report_line = f"Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}. Time: {current_time}\n"
    f.write(report_line)
    f.flush()

    print('...Saving Checkpoint...')
    actor.save_weights(actor_checkpoint_file)
    critic.save_weights(critic_checkpoint_file)
    tf.keras.backend.clear_session()

f.close()
