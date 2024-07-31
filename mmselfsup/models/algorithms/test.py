import torch


def align_featuremap(feature,random_order):

    ## accorading to mode, to align the feature map, for k_features


    if random_order == 0:  ###4321
        feature_1 = feature[:,:,0:5,0:5]
        feature_2 = feature[:,:,0:5,5:7]
        feature_3 = feature[:,:,5:7, 0:5]
        feature_4 = feature[:,:,5:7, 5:7]
    elif random_order == 1:  ###4231
        feature_1 = feature[:,:,0:5,0:5]
        feature_2 = feature[:,:,0:5,5:7]
        feature_3 = feature[:,:,5:7, 0:5]
        feature_4 = feature[:,:,5:7, 5:7]
    elif random_order == 2:  ### 3412
        feature_1 = feature[:, :, 0:5, 0:2]
        feature_2 = feature[:, :, 0:5, 2:7]
        feature_3 = feature[:, :, 5:7, 5:7]
        feature_4 = feature[:, :, 5:7, 0:5]
    elif random_order == 3:  ###2143
        feature_1 = feature[:, :, 0:2, 0:5]
        feature_2 = feature[:, :, 0:2, 5:7]
        feature_3 = feature[:, :, 2:7, 0:5]
        feature_4 = feature[:, :, 2:7, 5:7]
    elif random_order == 4:  ### 1324
        feature_1 = feature[:, :, 0:5, 0:5]
        feature_2 = feature[:, :, 0:5, 5:7]
        feature_3 = feature[:, :, 5:7, 0:5]
        feature_4 = feature[:, :, 5:7, 5:7]
    elif random_order == 5:  ### 3142
        feature_1 = feature[:, :, 0:5, 0:2]
        feature_2 = feature[:, :, 0:5, 2:7]
        feature_3 = feature[:, :, 5:7, 0:2]
        feature_4 = feature[:, :, 5:7, 2:7]
    elif random_order == 6:  ### 4132
        feature_1 = feature[:, :, 0:2, 0:2]
        feature_2 = feature[:, :, 0:5, 2:7]
        feature_3 = feature[:, :, 2:7, 0:2]
        feature_4 = feature[:, :, 5:7, 2:7]
    elif random_order == 7:  ### 1432
        feature_1 = feature[:, :, 0:5, 0:5]
        feature_2 = feature[:, :, 0:2, 5:7]
        feature_3 = feature[:, :, 5:7, 0:5]
        feature_4 = feature[:, :, 2:7, 5:7]

    print('f1',feature_1.size())
    print('f2', feature_2.size())
    print('f3', feature_3.size())
    print('f4', feature_4.size())


    return feature_1,feature_2,feature_3,feature_4


### test
# a = torch.tensor([[0,1],
#                   [2,3]])
#
# b = torch.tensor([[6,7],[8,9]])
# c = torch.tensor([[10,11],
#                   [12,13]])
#
# d = torch.tensor([[16,17],[18,19]])
#
# k = torch.cat((a,b,c,d),dim=0)
# g = k.view(2,2,2,2)
#
#
# print(g[1][0])


# a = torch.ones(16,3,224,224)
# b = torch.randn(16,3,224,224)
# c = a*b


### test

tensor1 = torch.ones((16,128))
tensor2 = torch.randn((16,128))
tensor3 = torch.randn((16,128))
tensor4 = torch.randn((16,128))

tensor_q = torch.cat((tensor1,tensor2,tensor3,tensor4),dim=0)
tensor_q_2 = tensor_q.view(-1,16,128).permute(1,2,0)

tensor5 = torch.randn((16,128))
tensor6 = torch.ones((16,128))
tensor7 = torch.randn((16,128))
tensor8 = torch.randn((16,128))

# ### similiar to densecl
# tensor9 = torch.ones((2,3))
# tensor10 = torch.randn((2,3))
# tensor11= torch.randn((2,3))
# tensor12= torch.randn((2,3))
#
# tensor_q = torch.cat((tensor9,tensor10,tensor11,tensor12),dim=0)
# tensor_q_2 = tensor_q.view(-1,2,3).permute(1,2,0)
# bn = 2
# tensor_q_3 = tensor_q_2.permute(2,0,1)
# tensor_q_pooling = tensor_q_3.view(2,2,bn,3)
# tensor_q_pooling = tensor_q_pooling.permute(2,3,0,1)
# print(tensor_q_pooling[:,:,0:1,1:2])


tensor_k = torch.cat((tensor5,tensor6,tensor7,tensor8),dim=0)
tensor_k_2 = tensor_k.view(-1,16,128).permute(1,2,0)

sim_matrix = torch.matmul(tensor_q_2.permute(0,2,1),tensor_k_2)
# sim_matrix = torch.cosine_similarity(tensor_q_2.permute(0,2,1),tensor_k_2)
dense_sim_ind = sim_matrix.max(dim=2)[1] ###[1] denotes index [0] denotes value

index_k_grid = torch.gather(tensor_k_2,2,dense_sim_ind.unsqueeze(1).expand(-1,tensor_k_2.size(1),-1))
dense_sim_q = (tensor_q_2*index_k_grid).sum(1)  ### measure the sim between q and k

l_pos_dense = dense_sim_q.view(-1).unsqueeze(-1)
tensor_q_2 = tensor_q_2.permute(0,2,1)
tensor_q_2 = tensor_q_2.reshape(-1,tensor_q_2.size(2))

print(tensor_q_2)

# q_norm_list = [tensor1,tensor2,tensor3,tensor4]
# k_list = [tensor5,tensor6,tensor7,tensor8]

# k_t_list = []
# ### match
# for i in range(len(q_norm_list)):
#     q_i = q_norm_list[i]
#     sim_list = []
#     for j in range(len(k_list)):
#         k_j = k_list[j]
#         ##compute cos similarity
#         cos_similarity = torch.sum(torch.cosine_similarity(q_i, k_j))
#         sim_list.append(cos_similarity)
#     max_index = sim_list.index(max(sim_list))
#     print(max_index)
#     k_t_i = k_list[max_index]
#     k_t_list.append(k_t_i)
#
# k_t_1,k_t_2 = k_t_list[0:2]
# print(k_t_1)



