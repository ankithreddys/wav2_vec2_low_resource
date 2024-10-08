import pre_processing_read
import pre_processing_conversational
import BPE
import dataset_splitting



if __name__ == '__main__':
    #print('conversational started')
    #CONV_DATASET_WITHOUT_TIME,CONV_DATASET_WITHOUT_TIME = pre_processing_conversational.dataset_creation('/as/as_IN_SD/CONVERSATIONAL')
    #print("conversational completed")
    #READ_DATASET = pre_processing_read.dataset_creation('/as/as_IN_SD/READ')
    #print("read completed")
    BPE.vocab_extraction(1000)
    print("vocab completed")
    
    #dataset_splitting.split()
    #print("data splitting completed")
