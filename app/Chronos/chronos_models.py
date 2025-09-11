# ============================================================================
# Chronos 모델 정의 - 개선된 GRU+CNN 하이브리드 모델 (Chronos_analysis_fixed.py 반영)
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List

class AttentionLayer(nn.Module):
    """
    Attention 메커니즘을 구현한 레이어
    Feature importance를 계산하기 위해 사용됩니다.
    """
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size)
        attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)  # (batch_size, seq_len)
        
        # Weighted sum
        context_vector = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_size)
        
        return context_vector, attention_weights

class FeatureAttentionLayer(nn.Module):
    """
    Feature-level Attention 메커니즘
    각 피처의 중요도를 계산합니다.
    """
    def __init__(self, input_size: int):
        super(FeatureAttentionLayer, self).__init__()
        self.input_size = input_size
        self.feature_attention = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        
        # Feature attention weights
        feature_weights = torch.sigmoid(self.feature_attention(x))  # (batch_size, seq_len, input_size)
        
        # Apply attention
        attended_features = x * feature_weights
        
        # Average feature importance across sequence
        avg_feature_importance = torch.mean(feature_weights, dim=(0, 1))  # (input_size,)
        
        return attended_features, avg_feature_importance

class GRU_CNN_HybridModel(nn.Module):
    """
    개선된 GRU + CNN 하이브리드 모델 (Chronos_analysis_fixed.py 기반)
    사람별 시계열 패턴 학습에 최적화됨
    """
    def __init__(self, input_size: int, gru_hidden: int = 32, cnn_filters: int = 16, 
                 kernel_sizes: list = [2, 3], dropout: float = 0.2, num_classes: int = 2):
        super(GRU_CNN_HybridModel, self).__init__()
        
        self.input_size = input_size
        self.gru_hidden = gru_hidden
        self.cnn_filters = cnn_filters
        
        # GRU 브랜치
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=1,
            dropout=dropout,
            batch_first=True
        )
        self.gru_dropout = nn.Dropout(dropout)
        
        # CNN 브랜치
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_layer = nn.Sequential(
                nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(cnn_filters),
                nn.AdaptiveMaxPool1d(1)
            )
            self.conv_layers.append(conv_layer)
        
        # 분류기
        combined_features = gru_hidden + len(kernel_sizes) * cnn_filters
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 어텐션
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden, 1),
            nn.Softmax(dim=1)
        )
        
        # Temperature Scaling 파라미터 (극단값 문제 해결)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Store attention weights for interpretation
        self.last_temporal_attention = None
        self.last_feature_importance = None
        
    def forward(self, x):
        # GRU + 어텐션
        gru_out, _ = self.gru(x)
        attention_weights = self.attention(gru_out)
        gru_features = torch.sum(gru_out * attention_weights, dim=1)
        gru_features = self.gru_dropout(gru_features)
        
        # Store attention for interpretation
        self.last_temporal_attention = attention_weights.squeeze(-1)
        
        # CNN
        x_cnn = x.transpose(1, 2)
        cnn_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x_cnn).squeeze(-1)
            cnn_outputs.append(conv_out)
        
        cnn_features = torch.cat(cnn_outputs, dim=1)
        
        # 결합 및 분류
        combined_features = torch.cat([gru_features, cnn_features], dim=1)
        logits = self.classifier(combined_features)
        
        # Temperature Scaling 적용 (극단값 문제 해결)
        scaled_logits = logits / self.temperature
        
        return scaled_logits
    
    def get_attention_weights(self):
        """
        마지막 forward pass의 attention weights를 반환합니다.
        """
        return {
            'temporal_attention': self.last_temporal_attention,
            'feature_importance': self.last_feature_importance
        }
    
    def interpret_prediction(self, x, feature_names: List[str] = None):
        """
        예측에 대한 해석 정보를 제공합니다.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probabilities = F.softmax(output, dim=1)
            
            # Temporal attention 가져오기
            temporal_attention = self.last_temporal_attention.cpu().numpy() if self.last_temporal_attention is not None else None
            
            # 간단한 feature importance (gradient 기반)
            x.requires_grad_(True)
            output_for_grad = self.forward(x)
            grad = torch.autograd.grad(output_for_grad[:, 1].sum(), x, create_graph=False)[0]
            feature_importance = torch.mean(torch.abs(grad), dim=(0, 1)).cpu().numpy()
            feature_importance = feature_importance / np.sum(feature_importance)
            
            interpretation = {
                'predictions': output.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'feature_importance': feature_importance,
                'temporal_attention': temporal_attention,
                'feature_names': feature_names if feature_names else [f'Feature_{i}' for i in range(self.input_size)]
            }
            
            return interpretation

class ChronosModelTrainer:
    """
    Chronos 모델 학습을 위한 트레이너 클래스
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """
        한 에포크 학습
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        return total_loss / len(dataloader), correct / total
    
    def evaluate(self, dataloader, criterion):
        """
        모델 평가
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
            'predictions': all_predictions,
            'labels': all_labels
        }

def create_hybrid_model(input_size: int, **kwargs) -> GRU_CNN_HybridModel:
    """
    개선된 하이브리드 모델 생성 헬퍼 함수
    """
    return GRU_CNN_HybridModel(
        input_size=input_size,
        gru_hidden=kwargs.get('gru_hidden', 32),
        cnn_filters=kwargs.get('cnn_filters', 16),
        kernel_sizes=kwargs.get('kernel_sizes', [2, 3]),
        dropout=kwargs.get('dropout', 0.2),
        num_classes=kwargs.get('num_classes', 2)
    )

# 하위 호환성을 위한 별칭
def create_attention_model(input_size: int, **kwargs) -> GRU_CNN_HybridModel:
    """
    하위 호환성을 위한 별칭
    """
    return create_hybrid_model(input_size, **kwargs)
