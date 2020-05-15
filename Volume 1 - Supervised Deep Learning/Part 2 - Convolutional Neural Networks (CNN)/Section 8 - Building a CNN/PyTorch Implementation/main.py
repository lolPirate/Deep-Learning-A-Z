from Data import  Data
from Agent import Agent


if __name__ == '__main__':
    data_params = {'train_path':r'..\dataset\training_set','test_path':r'..\dataset\test_set','batch_size':32}

    data_set = Data(**data_params)

    agent_params = {'lr':1e-3,'epochs':50,'train_data':data_set.get_train_data(),'test_data':data_set.get_test_data()}

    agent = Agent(**agent_params)
    
    agent.learn()

    