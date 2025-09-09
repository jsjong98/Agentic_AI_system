# ============================================================================
# Chronos 모델 정의 - Attention 메커니즘 포함
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

class GRU_CNN_AttentionModel(nn.Module):
    """
    GRU + CNN + Attention 하이브리드 모델
    Feature importance와 시계열 attention을 모두 제공합니다.
    """
    def __init__(self, input_size: int, gru_hidden: int = 64, cnn_filters: int = 32, 
                 dropout: float = 0.3, num_classes: int = 2):
        super(GRU_CNN_AttentionModel, self).__init__()
        
        self.input_size = input_size
        self.gru_hidden = gru_hidden
        self.cnn_filters = cnn_filters
        
        # Feature-level attention
        self.feature_attention = FeatureAttentionLayer(input_size)
        
        # GRU layers
        self.gru1 = nn.GRU(input_size, gru_hidden, batch_first=True, dropout=dropout)
        self.gru2 = nn.GRU(gru_hidden, gru_hidden, batch_first=True, dropout=dropout)
        
        # Temporal attention
        self.temporal_attention = AttentionLayer(gru_hidden)
        
        # CNN layers for pattern detection
        self.conv1d = nn.Conv1d(in_channels=gru_hidden, out_channels=cnn_filters, 
                               kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(gru_hidden + cnn_filters, gru_hidden)
        self.fc2 = nn.Linear(gru_hidden, num_classes)
        
        # Store attention weights for interpretation
        self.last_temporal_attention = None
        self.last_feature_importance = None
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        
        # Feature-level attention
        x_attended, feature_importance = self.feature_attention(x)
        self.last_feature_importance = feature_importance
        
        # GRU processing
        gru_out1, _ = self.gru1(x_attended)
        gru_out2, _ = self.gru2(gru_out1)
        
        # Temporal attention
        attended_output, temporal_weights = self.temporal_attention(gru_out2)
        self.last_temporal_attention = temporal_weights
        
        # CNN processing
        cnn_input = gru_out2.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        cnn_out = F.relu(self.conv1d(cnn_input))
        cnn_pooled = self.pool(cnn_out).squeeze(-1)  # (batch_size, cnn_filters)
        
        # Combine GRU attention output and CNN output
        combined = torch.cat([attended_output, cnn_pooled], dim=1)
        
        # Classification
        x = self.dropout(combined)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output
    
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
            
            attention_info = self.get_attention_weights()
            
            # Feature importance 정규화
            feature_importance = attention_info['feature_importance'].cpu().numpy()
            feature_importance = feature_importance / np.sum(feature_importance)
            
            # Temporal attention 정규화
            temporal_attention = attention_info['temporal_attention'].cpu().numpy()
            
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

def create_attention_model(input_size: int, **kwargs) -> GRU_CNN_AttentionModel:
    """
    Attention 모델 생성 헬퍼 함수
    """
    return GRU_CNN_AttentionModel(
        input_size=input_size,
        gru_hidden=kwargs.get('gru_hidden', 64),
        cnn_filters=kwargs.get('cnn_filters', 32),
        dropout=kwargs.get('dropout', 0.3),
        num_classes=kwargs.get('num_classes', 2)
    )
