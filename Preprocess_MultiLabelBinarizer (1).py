
# coding: utf-8

# In[37]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
import os
import h5py
import json
import h5py


# In[17]:


distance_data_path = "data.csv"
hnsw_result_path = "/home/lab4/code/HNSW/KNN-Evaluate/hnsw_result1111.h5py"
test_file_path = "test_image_feature.csv"
train_file_path = "vect_itemid130k.h5py"


# ### Loading hnsw data

# In[18]:


f = h5py.File(hnsw_result_path) 
hnsw_ids_result = np.array(f["itemID"])


# ### Loading testing data and sort() test_ids to match to result_hnsw_ids
# 

# In[19]:


df = pd.read_csv(test_file_path, sep="\t", converters={1: json.loads}).reset_index()
df.columns = ["item_id", "vect"]
test_vects = np.array(df.vect.tolist())
test_ids = np.array(df.item_id.tolist())
test_ids = sorted(test_ids)


# ### Load train_id list to make list fit() 

# In[43]:


h5 = h5py.File(train_file_path)
train_item_fit = np.array(h5["item"])


# In[44]:


len(train_item_fit)


# ### Loading KNN distance reference

# In[103]:



df_distance_data = pd.read_csv(distance_data_path, iterator=True, chunksize=500000)
print("loading ... done ")


# In[114]:


knn_matrix = []
cases = 10
for j in range(cases):  
    print(j)
    df_tmp = pd.DataFrame(columns =["distance","sou_id","des_id"])
    list_id=[]
    for chunk in pd.read_csv(distance_data_path, iterator=True, chunksize=500000):
        chunk.columns =["distance","sou_id","des_id"]
        df_tmp = df_tmp.append(chunk[chunk["sou_id"] == test_ids[j]])
    print("df_tmp before sort",df_tmp.shape)
    df_tmp=df_tmp.sort_values(by=['distance'])
    print("df_tmp after sort",df_tmp.shape)
    list_id = list(df_tmp.des_id.values)[:200]
    print("len list_id: ", len(list_id))
    knn_matrix.append(list_id)
print("finished")


# In[115]:


np.array(knn_matrix).shape


# In[116]:


hnsw_matrix = hnsw_ids_result[:cases,:]


# In[117]:


def precision_recall_f1_support(ground_truth, prediction):
    """
    Compute the recall, precision and f1
        Ex:
        ground_truth:
        [[0, 0, 1, 1, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 1, 1, 1]]
        prediction:
        [[0, 1, 0, 1, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 1, 1, 1]]
    Parameters
    ----------
    ground_truth : array
                   An array that specifies which tags is annotated in ground_truth
    prediction : array
                An array that specifies which tags is marked as the prediction

    Returns
    -------
    u_precision : double
                The mean of precision across all items
    u_recall: double
            The mean of recall across all items
    u_f1: double
        F1 that specified by 2 * (precision * recall) / (precision + recall)
    """
    # Calculate the number of prediction labels and num_ground_truth labels
    num_prediction = np.count_nonzero(prediction, axis=1)
    num_ground_truth = np.count_nonzero(ground_truth, axis=1)

    # Calculate the number of true positive prediction
    num_true_positive_pred = np.count_nonzero(ground_truth & prediction, axis=1)

    # Calculate the recall, precision, f1 for each item
    precision = num_true_positive_pred / num_prediction
    recall = num_true_positive_pred / num_ground_truth
    f1 = 2 * (precision * recall) / (precision + recall)

    # Get mean
    u_precision = np.mean(precision)
    u_recall = np.mean(recall)
    u_f1 = np.mean(f1)

    return u_precision, u_recall, u_f1


# In[118]:


lb = preprocessing.MultiLabelBinarizer()


# In[119]:


lb.fit(train_item_fit)


# In[120]:


hnsw_matrix.shape


# In[121]:


knn_matrix


# In[122]:


len(lb.classes_)


# In[123]:


precision_recall_f1_support(lb.transform(np.array(knn_matrix)), lb.transform(hnsw_matrix))

