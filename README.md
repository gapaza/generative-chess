# Generative Chess

This project exists on two servers
1. An aws p3.2xlarge instance for pre-training
2. An aws ri7z.8xlarge instance for fine-tuning (cpu bound updates)






### Transferring Datasets
- scp -i ./keyname.pem -r /dataset/path ubuntu@3.22.42.241:/home/ubuntu/generative-chess/datasets



