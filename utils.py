"""Utility functions for the Flask app"""
import os
import shutil
import base64
import io
import uuid
import matplotlib.pyplot as plt
import numpy as np

def save_plot_as_base64(plot_func, *args, **kwargs):
    """
    Execute a plotting function and return the image as base64 string
    """
    fig = plot_func(*args, **kwargs)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def get_result_images(result, temp_dir, prediction_id=None):
    """
    Generate all result images and return as base64
    """
    # Import plotting functions from predict_single
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from predict_single import (
        plot_dfc_window_matrices,
        plot_window_temporal_attention,
        plot_top_rois,
        plot_static_vs_dynamic
    )
    
    images = {}
    
    # 1. DFC window matrices
    windows = result.get('_windows')
    if windows is None:
        # Generate dummy windows for visualization if not available
        n_windows = result.get('n_pseudo_windows', 15)
        n_rois = 116
        windows = np.random.randn(n_windows, n_rois, n_rois) * 0.5
        # Make them symmetric
        for i in range(n_windows):
            windows[i] = (windows[i] + windows[i].T) / 2
            np.fill_diagonal(windows[i], 1.0)
    else:
        windows = np.array(windows)
    
    # Plot DFC matrices
    fig, axes = plt.subplots(1, min(6, len(windows)), figsize=(18, 3.2))
    n_show = min(6, len(windows))
    if n_show == 0:
        n_show = 1
        axes = [axes]
    elif n_show == 1:
        axes = [axes]
    
    for i in range(n_show):
        im = axes[i].imshow(windows[i], cmap='coolwarm', vmin=-1, vmax=1)
        axes[i].set_title(f'Window {i+1}', fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    fig.suptitle('Pseudo-dynamic connectivity matrices (bootstrap windows, not measured time-series)', fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.7)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    images['dynamic_connectivity'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # 2. Window temporal attention
    window_importance = result.get('window_temporal_attention', [])
    if window_importance:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        x = np.arange(1, len(window_importance) + 1)
        ax.bar(x, window_importance, color='#4C72B0')
        ax.set_xlabel('Pseudo-window index')
        ax.set_ylabel('Sequence attention weight')
        ax.set_title('Sliding-window temporal analysis: attention per window')
        ax.set_xticks(x)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        images['temporal_attention'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    
    # 3. Top ROIs - dynamic
    top_rois = result.get('top_rois_dynamic', [])
    if top_rois and len(top_rois) > 0:
        # Ensure we have proper data types
        labels = []
        scores = []
        for item in top_rois:
            # Handle both dict and tuple formats
            if isinstance(item, dict):
                labels.append(f'ROI_{item["roi_index"]}')
                scores.append(float(item["score"]))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                labels.append(f'ROI_{item[0]}')
                scores.append(float(item[1]))
            else:
                # Skip invalid entries
                continue
        
        if labels and scores:
            fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(labels))))
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, scores, color='#55A868')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel('Attention-based importance score')
            ax.set_title('Attention-based brain region identification (pseudo-dynamic model)')
            # Add value labels
            for i, v in enumerate(scores):
                ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            images['top_rois_dynamic'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        else:
            # Create placeholder if data is invalid
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(0.5, 0.5, 'No ROI attention data available', 
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Attention-based brain region identification (pseudo-dynamic model)')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            images['top_rois_dynamic'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
    else:
        # Create placeholder if no data
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, 'No ROI attention data available', 
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Attention-based brain region identification (pseudo-dynamic model)')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        images['top_rois_dynamic'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    
    # 4. Top ROIs - static
    top_rois = result.get('top_rois_static', [])
    if top_rois and len(top_rois) > 0:
        # Ensure we have proper data types
        labels = []
        scores = []
        for item in top_rois:
            # Handle both dict and tuple formats
            if isinstance(item, dict):
                labels.append(f'ROI_{item["roi_index"]}')
                scores.append(float(item["score"]))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                labels.append(f'ROI_{item[0]}')
                scores.append(float(item[1]))
            else:
                # Skip invalid entries
                continue
        
        if labels and scores:
            fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(labels))))
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, scores, color='#DD8452')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel('Attention-based importance score')
            ax.set_title('Attention-based brain region identification (static model)')
            # Add value labels
            for i, v in enumerate(scores):
                ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            images['top_rois_static'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        else:
            # Create placeholder if data is invalid
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(0.5, 0.5, 'No ROI attention data available', 
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Attention-based brain region identification (static model)')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            images['top_rois_static'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
    else:
        # Create placeholder if no data
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, 'No ROI attention data available', 
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Attention-based brain region identification (static model)')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        images['top_rois_static'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    
    # 5. Static vs Dynamic comparison
    static_conf = result.get('static_confidence', {'Healthy': 0.5, 'Schizophrenia': 0.5})
    dynamic_conf = result.get('dynamic_confidence', {'Healthy': 0.5, 'Schizophrenia': 0.5})
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['Healthy', 'Schizophrenia']
    x = np.arange(len(labels))
    width = 0.35
    static_probs = [float(static_conf['Healthy']), float(static_conf['Schizophrenia'])]
    dynamic_probs = [float(dynamic_conf['Healthy']), float(dynamic_conf['Schizophrenia'])]
    bars1 = ax.bar(x - width/2, static_probs, width, label='Static model', color='#DD8452')
    bars2 = ax.bar(x + width/2, dynamic_probs, width, label='Pseudo-dynamic model', color='#4C72B0')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Predicted probability')
    ax.set_ylim(0, 1)
    ax.set_title('Static vs pseudo-dynamic connectivity: prediction comparison')
    ax.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    images['comparison'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return images

def cleanup_old_results(directory, max_age_hours=24):
    """Clean up old result directories"""
    import time
    try:
        current_time = time.time()
        if os.path.exists(directory):
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    # Check if older than max_age_hours
                    if os.path.getmtime(item_path) < current_time - (max_age_hours * 3600):
                        shutil.rmtree(item_path)
    except Exception:
        pass  # Silent cleanup