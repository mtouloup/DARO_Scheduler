
# Kubernetes MARL-Based Pod Scheduling (DARO Framework)

This repository implements a **Multi-Agent Reinforcement Learning (MARL) scheduler** for **Kubernetes clusters** using the **QMIX algorithm**. The system enables worker nodes to **bid for pod execution**, ensuring **efficient task distribution** based on available resources.

## Overview
This project **simulates Kubernetes pod scheduling** using **multi-agent reinforcement learning (MARL)**. Each **worker node is an agent**, and a **broker collects bids from agents** to assign tasks efficiently.

**Key Features:**
**Multi-Agent Reinforcement Learning (MARL)** using QMIX  
**Kubernetes-style scheduling** with CPU & memory constraints  
**Dynamic cluster scaling** (agents can increase/decrease)  
**Randomized bid-based pod scheduling**  
**Training on CPU/GPU automatically detected**  

---

## How It Works
1. A **Kubernetes pod request** (task) is generated with specific **CPU & memory requirements**.
2. Each **worker node (agent)** checks its **available resources** and **submits a bid** (1-10) if it can handle the task.
3. The **broker collects all bids** and **assigns the task** to the **highest bidder**.
4. A **reward system** ensures:
   - **Winning agent gets +1 reward**.
   - **Losing agents get -0.1 penalty**.
5. The model is trained using **QMIX**, a deep MARL approach.

---

## Installation
### ** Prerequisites**
Ensure you have **Python 3.8+** installed.  

### ** Install Dependencies**
```
pip install -r requirements.txt
```

---

## Usage

### ** Training the Model**
Train the MARL scheduler using:
```
python qmix_training.py
```
- **This will detect your GPU automatically** and use it if available.
- Model will be saved as **`qmix_trained_k8s.pth`**.

---

### ** Testing the Model**
Once trained, evaluate the model:
```
python qmix_test.py
```
- The test script **simulates scheduling** and prints **which node wins each task**.
- The model should correctly assign tasks to **the highest bidder**.

---
## Running the API (Locally)

1. **Start the Flask Application**:
   Run the Flask application to start the API server:
   ```bash
   python app.py
   ```
   By default, the API will be accessible at `http://127.0.0.1:5000/`.

2. **Access the Swagger UI**:
   Open your web browser and navigate to `http://127.0.0.1:5000/daro` to access the Swagger UI. This interface allows you to interact with the API endpoints and view documentation.

3. **Using the `/schedule` Endpoint**:
   The `/schedule` endpoint determines the most suitable node for deploying a pod based on the trained model. You can test this endpoint using the Swagger UI or by sending a POST request with the following JSON structure:
   ```json
   {
       "cpu_request": 4.0,
       "memory_request": 8.0,
       "node_states": [
           {"node_id": "node-0", "cpu_available": 110.0, "memory_available": 20.0},
           {"node_id": "node-1", "cpu_available": 222.0, "memory_available": 6.0},
           {"node_id": "node-2", "cpu_available": 312.0, "memory_available": 30.0}
       ]
   }
   ```
   The API will respond with the ID of the node selected for the pod deployment.

**Note**: Ensure that the trained model is available and properly loaded by the API to make accurate scheduling decisions.

Running the Api with Docker
===================

To simplify deployment, you can containerize and run the application using Docker.

Prerequisites
-------------

Ensure you have Docker installed. You can download it fromDocker's official website: https://docs.docker.com/get-docker/

Steps to Run
------------

1\. Clone the Repository:

   git clone https://github.com/mtouloup/DARO\_Scheduler.git  
   cd DARO\_Scheduler

2\. Build the Docker Image:

docker build -tdaro\_scheduler .

3\. Run the Container:

docker run -p5000:5000 daro\_scheduler

This command startsthe application inside a container and exposes it on port 5000. Adjust the port if needed.

Access the API
--------------

Open your browser and navigate to:  http://127.0.0.1:5000/daro
## 🛠️ Implementation Details
### ** Agents (Worker Nodes)**
- Each agent **manages its own resources (CPU & Memory)**.
- Agents **bid based on their available resources**.

### ** Broker (Task Allocator)**
- **Collects bids from all agents**.
- **Assigns the task to the highest bidder**.

### ** Reinforcement Learning (QMIX)**
- Uses **a centralized critic with decentralized execution**.
- **Training optimizes task allocation decisions over time**.

---

## Contributors
- **Marios Touloupou** –  marios.touloupou@cut.ac.cy
- **Syedmafooqul Shah** - syedmafooqul.shah@cut.ac.cy


Contributions to the DARO Framework are welcome. Please follow the standard git workflow - fork, clone, commit, push, and create pull requests.

---

## License
This project is licensed under the **Apache License**.

---

## Acknowledgments

The research leading to the results presented in this project has received funding from the European Union’s funded Project Hyper-AI under grant agreement no 101135982.



