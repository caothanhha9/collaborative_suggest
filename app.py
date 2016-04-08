import datetime
from utils import data_helpers


def main():
    now_time = datetime.datetime.now().isoformat()
    print 'start'
    print now_time
    file_path = './data/customer_saving_salary'
    # data_helpers.auto_gen_and_save_classification_data(file_path=file_path)
    data_helpers.plot_data(file_path=file_path)
    print now_time
    print 'finish'

if __name__ == '__main__':
    main()
