# Assignment 10: Fast API Implementation using Docker
- Deploy it with FastAPI on Dokcer
- Use VIT model on trained on mnist dataset and GPT model trained on Harry Potter dataset
- Send the 100 API request using EC2 AWS server and check the Avg response time

# Approach: 
1. Downloaded the pytorch scripted model of VIT and GPT.
2. Downloaded the sample mnist images and harry potter book 1 from internet.
3. Created the Fast API script seperately for GPT and VIT models.
4. Created the "pytest.py", which will request the API response on random images and text from GPT and VIT model.
5. Created a requirement file which has required packages for both GPT and VIT
6. Created run.sh file which will execute both "gpt_api.py" and "vit_api.py" together
7. Created dockerfile to build the images that will create fast api server for both gpt and vit models at 8080 and 8000 ports respectively

# Steps for Execution:
1. Build the docker image (fast_api) using below command.
```
docker build -t fast_api .
```
2. Run the docker container using the below command.
```
docker run -it -p 8000:8000 -p 8080:8080 fast_api
```

3. Run the following script to check the response time ( pass the arguments for no of text queries and no of images)
```
python3 pytest.py -text 100 -images 100
```

# Avg Response time on AWS EC2 server
- Created a AWS EC2 instance (t3a.medium, ubuntu, 32 GB) and configured the inbound security ports for 8000 and 8080

- Copy the code folder inside EC2 instances 
```
ssh ubuntu@<ip address>
scp -r <local folder location> ubuntu@<ip address>:<VM file location>
```

- Built and run the docker instances inside EC2 machine

- Executed the pytest script on local to check the average reponse time

```
No of Text queries - 100
 Total Execution Time - 224843.303ms
 Avg Execution Time   - 2248.433 ms

No of Images - 100
 Total Execution Time - 12234.898ms
 Avg Execution Time   - 122.349 ms
 ``` 

# Group Members:
- Anurag Mittal
- Aman Jaipuria