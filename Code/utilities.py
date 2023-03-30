import os

import torch
import torchvision.datasets as datasets

from facenet_pytorch import InceptionResnetV1
from general import change_type

ROOT = os.getcwd() 

def embedding_data():
	''' Save feature khuôn mặt crop đã trích đặc trưng 
	'''
	resnet = InceptionResnetV1(pretrained='vggface2').eval()
	name_list = [] 
	embedding_list = [] 
 
	dataset = datasets.ImageFolder('runs/crops') 
	for img_crop, idx in dataset:
		face = change_type(img_crop, PIL=True)    

		emb = resnet(face)

		embedding_list.append(emb.detach()) 
		name_list.append(idx)#idx_to_class[idx])

	data = [embedding_list, name_list]
	torch.save(data, ROOT + '/models/parameter/embedding/SVM_mask_test.pt') 


def search_center():
	''' Tìm tất cả các khoảng cách và vector trung bình của từng class
	'''
	data = torch.load(ROOT + '/models/parameter/embedding/vggface2.pt')
	embedding_list = data[0]
	name_list = data[1]

	average_knn = {}
	face_list = {name: [] for name in set(name_list)}

	for idx in range(len(embedding_list)):
		face_list[name_list[idx]].append(embedding_list[idx])
	
	for name in set(name_list):
		faces_tensor = torch.cat(face_list[name], dim=0) # Change to tensor
		average_knn[name] = torch.sum(faces_tensor, dim=0).div(len(face_list[name]))

	torch.save(average_knn, ROOT + '/models/parameter/KNN/vggface2.pth')