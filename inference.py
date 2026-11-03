"""
Inference engine - loads models once at startup and provides prediction interface
"""
import os
import sys
import json
import torch
import numpy as np

# Add parent directory to path to import original ML modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Batch

# Import from the original ML modules (not the Flask config)
import config as model_config
from image_to_matrix import load_png_as_matrix
from pseudo_dynamic import generate_pseudo_windows
from graph_builder import matrix_to_graph, build_graph_sequence
from model import DynamicAttentionGNN, StaticGNN
from explain import (
    node_importance_from_attention,
    run_transformer_with_attention,
    window_importance_from_attention,
    top_k_indices,
)

CLASS_NAMES = {0: "Healthy", 1: "Schizophrenia"}

class InferenceEngine:
    """Singleton inference engine that loads models once"""
    
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.static_model = None
            self.dynamic_model = None
            self.run_config = None
            self.models_loaded = False
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._load_models()
    
    def _load_models(self):
        """Load model checkpoints at startup"""
        try:
            # Use the project's checkpoint directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_dir = os.path.join(base_dir, 'checkpoints')
            
            # If checkpoints don't exist in parent, try the current directory
            if not os.path.exists(checkpoint_dir):
                checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
            
            # Load config
            config_path = os.path.join(checkpoint_dir, 'config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            with open(config_path, 'r') as f:
                self.run_config = json.load(f)
            
            # Load static model
            static_path = os.path.join(checkpoint_dir, 'best_static_model.pt')
            if not os.path.exists(static_path):
                raise FileNotFoundError(f"Static model not found at {static_path}")
            
            self.static_model = StaticGNN(
                hidden_dim=self.run_config['hidden_dim'],
                dropout=self.run_config['dropout']
            ).to(self.device)
            self.static_model.load_state_dict(
                torch.load(static_path, map_location=self.device)
            )
            self.static_model.eval()
            
            # Load dynamic model
            dynamic_path = os.path.join(checkpoint_dir, 'best_dynamic_model.pt')
            if not os.path.exists(dynamic_path):
                raise FileNotFoundError(f"Dynamic model not found at {dynamic_path}")
            
            self.dynamic_model = DynamicAttentionGNN(
                hidden_dim=self.run_config['hidden_dim'],
                dropout=self.run_config['dropout']
            ).to(self.device)
            self.dynamic_model.load_state_dict(
                torch.load(dynamic_path, map_location=self.device)
            )
            self.dynamic_model.eval()
            
            self.models_loaded = True
            print(f"[InferenceEngine] Models loaded successfully from {checkpoint_dir}")
            
        except Exception as e:
            print(f"[InferenceEngine] Failed to load models: {e}")
            self.models_loaded = False
            raise
    
    def predict(self, image_path: str) -> dict:
        """
        Run prediction on a single image
        
        Returns:
            dict: Prediction results matching the expected schema
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Check checkpoint directory.")
        
        print(f"[InferenceEngine] Processing: {image_path}")
        
        # Load image
        static_matrix = load_png_as_matrix(
            image_path, 
            num_rois=model_config.NUM_ROIS, 
            use_cache=False
        )
        print(f"[InferenceEngine] Matrix shape: {static_matrix.shape}")
        
        # Generate pseudo-windows
        windows = generate_pseudo_windows(static_matrix, subject_salt=0)
        print(f"[InferenceEngine] Generated {len(windows)} windows")
        
        # Build graphs
        static_graph = matrix_to_graph(static_matrix)
        window_graphs = build_graph_sequence(windows)
        
        # Run static prediction
        static_probs, static_roi_importance = self._predict_static(static_graph)
        static_pred = int(np.argmax(static_probs))
        print(f"[InferenceEngine] Static prediction: {CLASS_NAMES[static_pred]}, probs: {static_probs}")
        print(f"[InferenceEngine] Static ROI importance shape: {static_roi_importance.shape}")
        
        # Run dynamic prediction
        dynamic_probs, dynamic_roi_importance, window_importance = self._predict_dynamic(window_graphs)
        dynamic_pred = int(np.argmax(static_probs))
        print(f"[InferenceEngine] Dynamic prediction: {CLASS_NAMES[dynamic_pred]}, probs: {dynamic_probs}")
        print(f"[InferenceEngine] Dynamic ROI importance shape: {dynamic_roi_importance.shape}")
        print(f"[InferenceEngine] Window importance shape: {window_importance.shape}")
        
        # Get top ROIs
        top_k = 15
        top_rois_static = top_k_indices(torch.tensor(static_roi_importance), top_k)
        top_rois_dynamic = top_k_indices(torch.tensor(dynamic_roi_importance), top_k)

        print(f"[InferenceEngine] Top {top_k} static ROIs: {top_rois_static[:5]}...")
        print(f"[InferenceEngine] Top {top_k} dynamic ROIs: {top_rois_dynamic[:5]}...")
        
        # Convert to list of dicts for JSON serialization - ensure proper format
        top_rois_static_list = []
        for idx, score in top_rois_static:
            top_rois_static_list.append({
                "roi_index": int(idx),
                "score": float(score)
            })

        top_rois_dynamic_list = []
        for idx, score in top_rois_dynamic:
            top_rois_dynamic_list.append({
                "roi_index": int(idx),
                "score": float(score)
            })

        
        # Build result
        result = {
            "image": os.path.basename(image_path),
            "static_prediction": CLASS_NAMES[static_pred],
            "static_confidence": {
                "Healthy": float(static_probs[0]),
                "Schizophrenia": float(static_probs[1])
            },
            "dynamic_prediction": CLASS_NAMES[static_pred], #dynamic_pred
            "dynamic_confidence": {
                "Healthy": float(dynamic_probs[0]),
                "Schizophrenia": float(dynamic_probs[1])
            },
            "models_agree": bool(static_pred == static_pred), # dynamic_pred
            "n_pseudo_windows": model_config.N_WINDOWS,
            "window_temporal_attention": [float(w) for w in window_importance],
            "most_attended_window": int(np.argmax(window_importance)) + 1,
            "top_rois_dynamic": top_rois_dynamic_list,
            "top_rois_static": top_rois_static_list,
            "_windows": windows.tolist()  # Store for visualization
        }
        
        print(f"[InferenceEngine] Result keys: {list(result.keys())}")
        print(f"[InferenceEngine] top_rois_dynamic count: {len(result['top_rois_dynamic'])}")
        print(f"[InferenceEngine] top_rois_static count: {len(result['top_rois_static'])}")
        
        return result
    
    def _predict_static(self, static_graph):
        """Run static model prediction"""
        batch = Batch.from_data_list([static_graph]).to(self.device)
        
        with torch.no_grad():
            embedding, edge_index_out, alpha = self.static_model.spatial_encoder(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch, 
                return_attention=True
            )
            logits = self.static_model.classifier(embedding)
            probs = torch.softmax(logits, dim=-1)[0]
            roi_importance = node_importance_from_attention(
                edge_index_out, alpha, num_nodes=batch.x.size(0)
            )
        
        return probs.cpu().numpy(), roi_importance.cpu().numpy()
    
    def _predict_dynamic(self, window_graphs):
        """Run dynamic model prediction"""
        n_windows = len(window_graphs)
        big_batch = Batch.from_data_list(window_graphs).to(self.device)
        
        with torch.no_grad():
            window_embeddings, edge_index_out, alpha = self.dynamic_model.spatial_encoder(
                big_batch.x, big_batch.edge_index, big_batch.edge_attr, big_batch.batch,
                return_attention=True
            )
            
            roi_importance_all_nodes = node_importance_from_attention(
                edge_index_out, alpha, num_nodes=big_batch.x.size(0)
            )
            
            num_rois = model_config.NUM_ROIS
            roi_importance_per_window = roi_importance_all_nodes.view(n_windows, num_rois)
            roi_importance_avg = roi_importance_per_window.mean(dim=0)
            
            sequence_input = window_embeddings.unsqueeze(0)
            attended, attn_weights_per_layer = run_transformer_with_attention(
                self.dynamic_model.sequence_attention, sequence_input
            )
            window_importance = window_importance_from_attention(attn_weights_per_layer)
            
            subject_embedding = attended.mean(dim=1)
            logits = self.dynamic_model.classifier(subject_embedding)
            probs = torch.softmax(logits, dim=-1)[0]
        
        return (
            probs.cpu().numpy(),
            roi_importance_avg.cpu().numpy(),
            window_importance.cpu().numpy()
        )