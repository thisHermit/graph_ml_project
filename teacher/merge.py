import torch
import os

kd = torch.load(os.path.join(os.getcwd(), "teacher_results/node-embeddings.pt"))
lg = torch.load(os.path.join(os.getcwd(), "teacher_results/logits-embeddings.pt"))
kd["ptr"] = kd["ptr"].long()
kd["logits"] = lg["logits"]
torch.save(kd, os.path.join(os.getcwd(), "teacher_results/teacher-knowledge.pt"))

print("Teacher knowledge saved to teacher_results/teacher-knowledge.pt")
os.remove(os.path.join(os.getcwd(), "teacher_results/node-embeddings.pt"))
os.remove(os.path.join(os.getcwd(), "teacher_results/logits-embeddings.pt"))

print("Intermediate files removed")
