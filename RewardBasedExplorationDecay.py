""" inspired from https://gym.openai.com/evaluations/eval_lEi8I8v2QLqEgzBxcvRIaA/ """

import os
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import gym
from collections import deque

class EpisodicAgent(object):
    """
    Episodic agent is a simple nearest-neighbor based agent:
    - At training time it remembers all tuples of (state, action, reward).
    - After each episode it computes the empirical value function based
        on the recorded rewards in the episode.
    - At test time it looks up k-nearest neighbors in the state space
        and takes the action that most often leads to highest average value.
    """

    def __init__(self, action_space, summary_writer, mode):
        self.action_space = action_space
        #assert isinstance(action_space, gym.spaces.discrete.Discrete), 'unsupported action space for now.'

        # options
        self.epsilon = 1.0  # probability of choosing a random action
        self.epsilon_decay = 0.98  # decay of epsilon per episode
        self.epsilon_min = 0
        self.nnfind = 500  # how many nearest neighbors to consider in the policy?
        self.mem_needed = 500  # amount of data to have before we can start exploiting
        self.mem_size = 50000  # maximum size of memory
        self.gamma = 0.95  # discount factor

        # internal vars
        self.iter = 0
        self.mem_pointer = 0  # memory pointer
        self.max_pointer = 0
        self.db = None  # large array of states seen
        self.dba = {}  # actions taken
        self.dbr = {}  # rewards obtained at all steps
        self.dbv = {}  # value function at all steps, computed retrospectively
        self.ep_start_pointer = 0

        # Tensorboard
        self.writer = summary_writer

        # RBED exclusive
        self.reward_threshold = 0  # Keep track of reward target for the agent
        self.epsilon_max = 1.0
        self.steps_to_move_in = 200  # Set to target value.
        self.quanta = (self.epsilon_max - self.epsilon_min) / self.steps_to_move_in  # what value to move epsilon by
        self.target_increment = 1  # How much to increment the target every time the agent meets it.

        self.mode = mode


    def exp_decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def rb_decay_epsilon(self, curretnt_reward = 0):
        if curretnt_reward > self.reward_threshold:
            self.reward_threshold += self.target_increment
            self.epsilon -= self.quanta



    def act(self, observation, reward, done, last_total_reward=0, iteration=0):
        assert isinstance(observation, np.ndarray) and observation.ndim == 1, 'unsupported observation type for now.'

        if self.db is None:
            # lazy initialization of memory
            self.db = np.zeros((self.mem_size, observation.size))
            self.mem_pointer = 0
            self.ep_start_pointer = 0

        # we have enough data, we want to explore, and we have seen at least one episode already (so values were computed)
        if self.iter > self.mem_needed and np.random.rand() > self.epsilon and self.dbv:
            # exploit: find the few closest states and pick the action that led to highest rewards
            # 1. find k nearest neighbors
            ds = np.sum((self.db[:self.max_pointer] - observation) ** 2, axis=1)  # L2 distance
            ix = np.argsort(ds)  # sorts ascending by distance
            ix = ix[:min(len(ix), self.nnfind)]  # crop to only some number of nearest neighbors

            # find the action that leads to most success. do a vote among actions
            adict = {}
            ndict = {}
            for i in ix:
                vv = self.dbv[i]
                aa = self.dba[i]
                vnew = adict.get(aa, 0) + vv
                adict[aa] = vnew
                ndict[aa] = ndict.get(aa, 0) + 1

            for a in adict:  # normalize by counts
                adict[a] = adict[a] / ndict[a]

            its = [(y, x) for x, y in adict.items()]
            its.sort(reverse=True)  # descending
            a = its[0][1]

        else:
            # explore: do something random
            a = self.action_space.sample()

        # record move to database
        if self.mem_pointer < self.mem_size:
            self.db[self.mem_pointer] = observation  # save the state
            self.dba[self.mem_pointer] = a  # and the action we took
            self.dbr[self.mem_pointer - 1] = reward  # and the reward we obtained last time step
            self.dbv[self.mem_pointer - 1] = 0
        self.mem_pointer += 1
        self.iter += 1

        if done:  # episode Ended;

            # compute the estimate of the value function based on this rollout
            v = 0
            for t in reversed(range(self.ep_start_pointer, self.mem_pointer)):
                v = self.gamma * v + self.dbr.get(t, 0)
                self.dbv[t] = v

            self.ep_start_pointer = self.mem_pointer
            self.max_pointer = min(max(self.max_pointer, self.mem_pointer), self.mem_size)

            if self.mode == 'EXPONENTIAL':
                self.exp_decay_epsilon()

            elif self.mode == 'RBED':
                self.rb_decay_epsilon(last_total_reward)

            self.writer.add_scalar('epsilon', self.epsilon, iteration)

        return a


class RecordPanda(object):

    def __init__(self, header, root_path='records',  mode='unknown'):
        self.header = header
        self.root_path = root_path
        self.mode = mode

        self.pkl_path = self.root_path+"/df.pkl"
        self.csv_path =  self.root_path+"/df.csv"
        self.fthr_path = self.root_path + "/df.fthr"

        self.pkl100_path = self.root_path + "/df100.pkl"
        self.csv100_path = self.root_path + "/df100.csv"
        self.fthr100_path = self.root_path + "/df100.fthr"

        self.records = pd.DataFrame(columns=self.header)
        self.current = pd.Series(index=self.header).astype('object')

        self.records100 = pd.DataFrame(columns=self.header)
        self.current100 = pd.Series(index=self.header).astype('object')

        if os.path.exists(self.fthr_path):
            self.load_records()


    def save_records(self, to_csv = False):
        if to_csv:
            self.records.to_csv(self.csv_path)
            self.records100.to_csv(self.csv100_path)

        self.records.to_feather(self.fthr_path)
        self.records100.to_feather(self.fthr100_path)

    def load_records(self, from_csv = False):
        if from_csv:
            self.records = pd.read_csv(self.csv_path)
            self.records100 = pd.read_csv(self.csv100_path)
        else:
            self.records = pd.read_feather(self.fthr_path)
            self.records100 = pd.read_feather(self.fthr100_path)

    def add_to_record(self):
        self.records = self.records.append(self.current, ignore_index = True)
        self.current = pd.Series(index=self.header).astype('object')

        self.records100 = self.records100.append(self.current100, ignore_index = True)
        self.current100 = pd.Series(index=self.header).astype('object')

        self.save_records()


if __name__ == '__main__':

    #tensorboard
    MODE_E = 'EXPONENTIAL'
    MODE_R = 'RBED'
    MODE = MODE_E
    # TODO : Support folder for different modes.

    header = ['0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
        '22',
        '23',
        '24',
        '25',
        '26',
        '27',
        '28',
        '29',
        '30',
        '31',
        '32',
        '33',
        '34',
        '35',
        '36',
        '37',
        '38',
        '39',
        '40',
        '41',
        '42',
        '43',
        '44',
        '45',
        '46',
        '47',
        '48',
        '49',
        '50',
        '51',
        '52',
        '53',
        '54',
        '55',
        '56',
        '57',
        '58',
        '59',
        '60',
        '61',
        '62',
        '63',
        '64',
        '65',
        '66',
        '67',
        '68',
        '69',
        '70',
        '71',
        '72',
        '73',
        '74',
        '75',
        '76',
        '77',
        '78',
        '79',
        '80',
        '81',
        '82',
        '83',
        '84',
        '85',
        '86',
        '87',
        '88',
        '89',
        '90',
        '91',
        '92',
        '93',
        '94',
        '95',
        '96',
        '97',
        '98',
        '99',
        '100',
        '101',
        '102',
        '103',
        '104',
        '105',
        '106',
        '107',
        '108',
        '109',
        '110',
        '111',
        '112',
        '113',
        '114',
        '115',
        '116',
        '117',
        '118',
        '119',
        '120',
        '121',
        '122',
        '123',
        '124',
        '125',
        '126',
        '127',
        '128',
        '129',
        '130',
        '131',
        '132',
        '133',
        '134',
        '135',
        '136',
        '137',
        '138',
        '139',
        '140',
        '141',
        '142',
        '143',
        '144',
        '145',
        '146',
        '147',
        '148',
        '149',
        '150',
        '151',
        '152',
        '153',
        '154',
        '155',
        '156',
        '157',
        '158',
        '159',
        '160',
        '161',
        '162',
        '163',
        '164',
        '165',
        '166',
        '167',
        '168',
        '169',
        '170',
        '171',
        '172',
        '173',
        '174',
        '175',
        '176',
        '177',
        '178',
        '179',
        '180',
        '181',
        '182',
        '183',
        '184',
        '185',
        '186',
        '187',
        '188',
        '189',
        '190',
        '191',
        '192',
        '193',
        '194',
        '195',
        '196',
        '197',
        '198',
        '199',
        '200',
        '201',
        '202',
        '203',
        '204',
        '205',
        '206',
        '207',
        '208',
        '209',
        '210',
        '211',
        '212',
        '213',
        '214',
        '215',
        '216',
        '217',
        '218',
        '219',
        '220',
        '221',
        '222',
        '223',
        '224',
        '225',
        '226',
        '227',
        '228',
        '229',
        '230',
        '231',
        '232',
        '233',
        '234',
        '235',
        '236',
        '237',
        '238',
        '239',
        '240',
        '241',
        '242',
        '243',
        '244',
        '245',
        '246',
        '247',
        '248',
        '249',
        '250',
        '251',
        '252',
        '253',
        '254',
        '255',
        '256',
        '257',
        '258',
        '259',
        '260',
        '261',
        '262',
        '263',
        '264',
        '265',
        '266',
        '267',
        '268',
        '269',
        '270',
        '271',
        '272',
        '273',
        '274',
        '275',
        '276',
        '277',
        '278',
        '279',
        '280',
        '281',
        '282',
        '283',
        '284',
        '285',
        '286',
        '287',
        '288',
        '289',
        '290',
        '291',
        '292',
        '293',
        '294',
        '295',
        '296',
        '297',
        '298',
        '299',
        '300',
        '301',
        '302',
        '303',
        '304',
        '305',
        '306',
        '307',
        '308',
        '309',
        '310',
        '311',
        '312',
        '313',
        '314',
        '315',
        '316',
        '317',
        '318',
        '319',
        '320',
        '321',
        '322',
        '323',
        '324',
        '325',
        '326',
        '327',
        '328',
        '329',
        '330',
        '331',
        '332',
        '333',
        '334',
        '335',
        '336',
        '337',
        '338',
        '339',
        '340',
        '341',
        '342',
        '343',
        '344',
        '345',
        '346',
        '347',
        '348',
        '349',
        '350',
        '351',
        '352',
        '353',
        '354',
        '355',
        '356',
        '357',
        '358',
        '359',
        '360',
        '361',
        '362',
        '363',
        '364',
        '365',
        '366',
        '367',
        '368',
        '369',
        '370',
        '371',
        '372',
        '373',
        '374',
        '375',
        '376',
        '377',
        '378',
        '379',
        '380',
        '381',
        '382',
        '383',
        '384',
        '385',
        '386',
        '387',
        '388',
        '389',
        '390',
        '391',
        '392',
        '393',
        '394',
        '395',
        '396',
        '397',
        '398',
        '399',
        '400',
        '401',
        '402',
        '403',
        '404',
        '405',
        '406',
        '407',
        '408',
        '409',
        '410',
        '411',
        '412',
        '413',
        '414',
        '415',
        '416',
        '417',
        '418',
        '419',
        '420',
        '421',
        '422',
        '423',
        '424',
        '425',
        '426',
        '427',
        '428',
        '429',
        '430',
        '431',
        '432',
        '433',
        '434',
        '435',
        '436',
        '437',
        '438',
        '439',
        '440',
        '441',
        '442',
        '443',
        '444',
        '445',
        '446',
        '447',
        '448',
        '449',
        '450',
        '451',
        '452',
        '453',
        '454',
        '455',
        '456',
        '457',
        '458',
        '459',
        '460',
        '461',
        '462',
        '463',
        '464',
        '465',
        '466',
        '467',
        '468',
        '469',
        '470',
        '471',
        '472',
        '473',
        '474',
        '475',
        '476',
        '477',
        '478',
        '479',
        '480',
        '481',
        '482',
        '483',
        '484',
        '485',
        '486',
        '487',
        '488',
        '489',
        '490',
        '491',
        '492',
        '493',
        '494',
        '495',
        '496',
        '497',
        '498',
        '499',
        'Experiment',
        'Mode',
        'Solved',
    ]
    i_experiment = 500
    i_mode = 501
    i_solved = 502

    rp = RecordPanda(header = header, mode=MODE)
    # rp.current[i_mode] = MODE
    # rp.current100[i_mode] = MODE
    env = gym.make('CartPole-v0')

    # writer = SummaryWriter(comment=MODE)
    last_ep_reward = 0
    scores = deque(maxlen=100)  # because solution depends on last 100 solution.
    scores.append(int(0))



    run_count = 2 # how many times to repeat the experiment
    episode_count = 500
    max_steps = 200
    reward = 0
    done = False
    sum_reward_running = 0
    last_ep_reward = 0
    writer = SummaryWriter(comment=MODE)

    for r in trange(run_count, desc='RUN'):

        #print('run', (r+1), '/', run_count)
        rp.current[i_mode] = MODE
        rp.current100[i_mode] = MODE
        writer = SummaryWriter(comment=MODE)
        agent = EpisodicAgent(env.action_space, summary_writer=writer, mode=MODE)
        last_ep_reward = 0
        scores = deque(maxlen=100)  # because solution depends on last 100 solution.
        scores.append(int(0))

        experiment_name = 'abcdef' #TODO : get thensorboard name here
        rp.current[i_experiment] = experiment_name
        rp.current100[i_experiment] = experiment_name

        for i in trange(episode_count, desc='episode', ascii=True):
            ob = env.reset()
            sum_reward = 0

            for j in trange(max_steps, desc='step', ascii=True):
                action = agent.act(ob, reward, done, last_total_reward=last_ep_reward, iteration=i)
                ob, reward, done, _ = env.step(action)
                sum_reward += reward
                if done:
                    break

            last_ep_reward = sum_reward

            writer.add_scalar('reward', last_ep_reward, i)
            scores.append(int(last_ep_reward))
            current_score = sum(scores) / 100
            writer.add_scalar('last100', current_score, i)

            rp.current100[i] = current_score
            rp.current[i] = last_ep_reward

            if last_ep_reward >=195 and rp.current.isnull()[i_solved]:
                rp.current[i_solved] = i

            if current_score >= 195 and rp.current100.isnull()[i_solved]:
                rp.current100[i_solved] = i

        rp.add_to_record()
        rp.save_records()
    rp.save_records(to_csv=True)