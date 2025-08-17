# 🎶 Mini Project: Docker Architecture for Music Genre Classification

#### Université de Tunis – École Nationale Supérieure d'Ingénieurs de Tunis (ENSIT)
- **Course**: Nouvelles Architectures
- **Submission Date**: December 12, 2024
- **Section**: 3GInfo

---

## 📂 Project Structure

This project implements a **Docker-based architecture** for music genre classification using **Flask microservices**, a **frontend application**, and **Jenkins CI/CD integration**.

````plaintext
project-root/
│
├── data/                       
│   └── gtzan/                  
│
├── services/
│   ├── SVM_service/            
│   │   ├── app.py              
│   │   ├── model/              
│   │   │   └── svm_model.pkl   
│   │   ├── Dockerfile          
│   │   └── requirements.txt    
│   │
│   ├── VGG19_service/          
│   │   ├── app.py              
│   │   ├── model/              
│   │   │   └── vgg19_model.h5  
│   │   ├── Dockerfile          
│   │   └── requirements.txt    
│
├── frontend/                   
│   ├── src/
│   │   ├── components/         
│   │   ├── App.js              
│   │   ├── index.js            
│   ├── public/
│   │   └── index.html          
│   ├── Dockerfile              
│   └── package.json            
│
├── orchestrator/               
│   ├── app.py                  
│   ├── Dockerfile              
│   └── requirements.txt        
│
├── jenkins/
│   ├── Dockerfile              
│   └── jenkinsfile             
│
├── docker-compose.yml          
├── README.md                   
└── report/                     
    └── rapport.pdf             
`````

---

## 📑 Content Details

* **data**: Contains the GTZAN dataset and test WAV files.
* **services/SVM\_service**: Flask API for classification using a pre-trained SVM model.
* **services/VGG19\_service**: Flask API for classification using a pre-trained VGG19 model.
* **frontend**: Web interface to interact with Flask services and display predictions.
* **orchestrator**: Handles requests and aggregates results from SVM and VGG19 services.
* **jenkins**: CI/CD pipeline setup with Jenkins.
* **docker-compose.yml**: Defines services, networks, and volumes for orchestration.

---

## 🚀 Deployment Instructions

1. **Build & Run containers**

```bash
docker-compose up --build
```

2. **Access the services**

* Frontend: `http://localhost:5000`
* SVM Service: `http://localhost:<svm_port>`
* VGG19 Service: `http://localhost:<vgg19_port>`
* Jenkins: `http://localhost:8080`

3. **CI/CD with Jenkins**

* Jenkins automatically builds, tests, and deploys services.

4. **Testing the app**

* Use Jenkins pipelines to validate API endpoints and model predictions.

---

## 📸 Screenshots

### Frontend Interface

<p align="center">
  <img src="screenshots/frontend-ui.png" width="70%">
</p>

### Jenkins Pipeline

<p align="center">
  <img src="screenshots/jenkins-pipeline.png" width="70%">
</p>

---

## 📝 Report

The final **10-page report** includes:

* Docker-based architecture explanation
* ML model selection & implementation (SVM & VGG19)
* Flask service design & integration
* Deployment steps
* Test results & evaluations
