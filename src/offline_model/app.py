from flask import Flask, request, jsonify, Blueprint
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import torch
import numpy as np
from qmix_agent import QMIX

# Flask app setup with Swagger UI
app = Flask(__name__, static_folder='static', static_url_path='/daro/static')
CORS(app)

# Swagger UI route
@app.route('/daro')
def serve_swagger_ui():
    return app.send_static_file('swaggerui/index.html')

# Create API blueprint
api_blueprint = Blueprint('api', __name__)
api = Api(api_blueprint, version="0.1", title='Kubernetes QMIX Scheduler',
          description="A MARL-based scheduling API for Kubernetes pods.")

app.register_blueprint(api_blueprint, url_prefix='/daro')

# Configure Swagger UI
api_doc_blueprint = Blueprint('api_doc', __name__, url_prefix='/daro/doc')
api_doc = Api(api_doc_blueprint, doc='/docs')
api_doc.add_namespace(api)
app.register_blueprint(api_doc_blueprint)

# Load trained QMIX model
MODEL_PATH = "qmix_trained_k8s.pth"
state_dim = 2  # Each agent's state consists of (CPU availability, Memory availability)
action_dim = 11  # Agents can bid between 1 and 10

def load_qmix_model(num_agents):
    """Load the trained QMIX model."""
    qmix_agent = QMIX(num_agents=num_agents, input_dim=state_dim, output_dim=action_dim)
    checkpoint = torch.load(MODEL_PATH)
    
    # Load model weights
    for i in range(len(qmix_agent.q_networks)):
        qmix_agent.q_networks[i].load_state_dict(checkpoint)
        qmix_agent.q_networks[i].eval()  # Set to evaluation mode
    
    return qmix_agent

# Define request model for Swagger UI
pod_request_model = api.model('PodRequest', {
    'cpu_request': fields.Float(required=True, description="CPU cores requested by the Pod", example=4.0),
    'memory_request': fields.Float(required=True, description="Memory requested by the Pod (GB)", example=8.0),
    'node_states': fields.List(fields.Nested(api.model('NodeState', {
        'node_id': fields.String(required=True, description="Unique ID of the worker node"),
        'cpu_available': fields.Float(required=True, description="Available CPU cores on the node"),
        'memory_available': fields.Float(required=True, description="Available memory on the node (GB)")
    })), required=True, description="Current state of Kubernetes worker nodes", example=[
        {"node_id": "node-0", "cpu_available": 110.0, "memory_available": 20.0},
        {"node_id": "node-1", "cpu_available": 222.0, "memory_available": 6.0},
        {"node_id": "node-2", "cpu_available": 312.0, "memory_available": 30.0}
    ])
})

# Define response model
pod_response_model = api.model('PodResponse', {
    'scheduled_node': fields.String(description="The ID of the node selected for scheduling"),
    'bids': fields.List(fields.Integer, description="List of bid values for each node")
})

@api.route('/schedule')
class SchedulePod(Resource):
    @api.expect(pod_request_model, validate=True)
    @api.response(200, "Success", pod_response_model)
    @api.response(400, "Invalid input")
    def post(self):
        """Schedule a Pod on the best Kubernetes node"""
        data = request.json

        # Extract task details
        cpu_request = data["cpu_request"]
        memory_request = data["memory_request"]
        node_states = data["node_states"]

        if not cpu_request or not memory_request or not node_states:
            return {"error": "Invalid input. Provide cpu_request, memory_request, and node_states."}, 400

        # Convert node states to structured format
        states = []
        node_ids = []
        for node in node_states:
            node_ids.append(node["node_id"])
            states.append([node["cpu_available"], node["memory_available"]])

        states = np.array(states, dtype=np.float32)
        num_agents = len(states)

        # Load QMIX model for the current number of agents
        qmix_agent = load_qmix_model(num_agents)

        # Select actions (bids) using the trained model (fully greedy, epsilon=0)
        actions = qmix_agent.select_actions(states, epsilon=0)

        # Find the node with the highest bid
        best_node_index = np.argmax(actions)
        best_node = node_ids[best_node_index]

        return {"scheduled_node": best_node, "bids": actions}

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
