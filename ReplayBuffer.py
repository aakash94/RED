import pandas as pd

class ReplayBuffer(object):

    def __init__(self, capacity):

        r_buff_header = ['state', 'action', 'next_state', 'reward', 'done']
        self.capacity = capacity
        self.header = r_buff_header
        self.buffer = pd.DataFrame(columns=self.header)

    def push(self, df_row):
        if self.__len__() == self.capacity:
            # Probably exceeded capacity
            # remove a row (probably 1st one) here
            self.buffer = self.buffer.iloc[1:]
        # add to dataframe here
        self.buffer = pd.concat([self.buffer, df_row])

    def insert(self, stateV, actonV, next_stateV, rewardV, doneV):
        # Initialise data to lists.
        data = [{self.header[0]: stateV,
                 self.header[1]: actonV,
                 self.header[2]: next_stateV,
                 self.header[3]: rewardV,
                 self.header[4]: doneV}]

        # Creates DataFrame.
        df = pd.DataFrame(data)
        self.push(df)

    def sample(self, batch_size=0):
        if batch_size == 0:
            return self.buffer
        else:
            return self.buffer.sample(batch_size)

    def __len__(self):
        return self.buffer.shape[0]