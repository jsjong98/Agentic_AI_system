import axios from 'axios';

// API 기본 설정 - Supervisor 포트로 변경
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5006';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5분 타임아웃 (최적화 작업이 오래 걸릴 수 있음)
  headers: {
    'Content-Type': 'application/json',
  },
});

// 요청 인터셉터
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API 요청: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API 요청 오류:', error);
    return Promise.reject(error);
  }
);

// 응답 인터셉터
apiClient.interceptors.response.use(
  (response) => {
    console.log(`API 응답: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API 응답 오류:', error);
    
    // 에러 메시지 표준화
    const errorMessage = error.response?.data?.error || 
                        error.message || 
                        '알 수 없는 오류가 발생했습니다.';
    
    return Promise.reject(new Error(errorMessage));
  }
);

export const apiService = {
  // 서버 상태 확인
  async checkHealth() {
    try {
      const response = await apiClient.get('/health');
      return response.data;
    } catch (error) {
      throw new Error('서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.');
    }
  },

  // 데이터 로드
  async loadData(filePath) {
    try {
      const response = await apiClient.post('/load_data', {
        file_path: filePath
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // 파일 업로드 (FormData 사용)
  async uploadFile(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await apiClient.post('/upload_file', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // 임계값 계산
  async calculateThresholds(scoreColumns = null) {
    try {
      const response = await apiClient.post('/calculate_thresholds', {
        score_columns: scoreColumns
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // 가중치 최적화
  async optimizeWeights(method = 'bayesian', params = {}) {
    try {
      const payload = {
        method,
        ...params
      };
      
      const response = await apiClient.post('/optimize_weights', payload);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // 개별 직원 예측
  async predictEmployee(scores) {
    try {
      const response = await apiClient.post('/predict_employee', {
        scores
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // 최적화 방법 비교
  async compareMethods(methods = ['grid', 'scipy']) {
    try {
      const response = await apiClient.post('/compare_methods', {
        methods
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // 현재 결과 조회
  async getResults() {
    try {
      const response = await apiClient.get('/api/results');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // 결과 내보내기
  async exportResults(format = 'csv', includeData = true) {
    try {
      const response = await apiClient.post('/export_results', {
        format,
        include_data: includeData
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // 파일 다운로드
  async downloadFile(filePath) {
    try {
      const response = await apiClient.get(`/download/${encodeURIComponent(filePath)}`, {
        responseType: 'blob'
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Integration API
  async integrateResults(uploadedFiles) {
    try {
      const response = await apiClient.post('/integration/integrate', {
        files: uploadedFiles
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  async generateReport(reportConfig) {
    try {
      const response = await apiClient.post('/integration/generate_report', reportConfig);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Supervisor API
  async startSupervisorSession(config) {
    try {
      const response = await apiClient.post('/supervisor/start_session', config);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  async runSupervisorAnalysis(sessionId, analysisConfig) {
    try {
      const response = await apiClient.post(`/supervisor/analyze/${sessionId}`, analysisConfig);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  async runAgentCollaboration(sessionId) {
    try {
      const response = await apiClient.post(`/supervisor/collaborate/${sessionId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  async synthesizeResults(sessionId) {
    try {
      const response = await apiClient.post(`/supervisor/synthesize/${sessionId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  async pauseSupervisorSession(sessionId) {
    try {
      const response = await apiClient.post(`/supervisor/pause/${sessionId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  async resumeSupervisorSession(sessionId) {
    try {
      const response = await apiClient.post(`/supervisor/resume/${sessionId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  async stopSupervisorSession(sessionId) {
    try {
      const response = await apiClient.post(`/supervisor/stop/${sessionId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // XAI API
  async analyzeFeatureImportance(config) {
    try {
      const response = await apiClient.post('/xai/feature_importance', config);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  async analyzeSHAP(config) {
    try {
      const response = await apiClient.post('/xai/shap', config);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  async analyzeLIME(config) {
    try {
      const response = await apiClient.post('/xai/lime', config);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  async generateDecisionTree(config) {
    try {
      const response = await apiClient.post('/xai/decision_tree', config);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  async generateCounterfactuals(config) {
    try {
      const response = await apiClient.post('/xai/counterfactuals', config);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // 배치 분석 API
  async analyzeBatch(batchConfig) {
    try {
      const response = await apiClient.post('/batch_analyze', batchConfig);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Integration 보고서 생성
  async generateIntegrationReport(reportConfig) {
    try {
      const response = await apiClient.post('/api/workers/integration/generate_report', reportConfig);
      return response.data;
    } catch (error) {
      throw error;
    }
  }
};

// 유틸리티 함수들
export const apiUtils = {
  // 에러 메시지 추출
  getErrorMessage(error) {
    if (error.response?.data?.error) {
      return error.response.data.error;
    }
    if (error.message) {
      return error.message;
    }
    return '알 수 없는 오류가 발생했습니다.';
  },

  // 파일 크기 포맷팅
  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  // 숫자 포맷팅
  formatNumber(num, decimals = 4) {
    if (typeof num !== 'number') return num;
    return num.toFixed(decimals);
  },

  // 백분율 포맷팅
  formatPercentage(num, decimals = 1) {
    if (typeof num !== 'number') return num;
    return (num * 100).toFixed(decimals) + '%';
  },

  // 위험도 레벨 색상 반환
  getRiskLevelColor(riskLevel) {
    switch (riskLevel) {
      case '안전군':
        return '#52c41a';
      case '주의군':
        return '#faad14';
      case '고위험군':
        return '#ff4d4f';
      default:
        return '#666';
    }
  },

  // 위험도 레벨 아이콘 반환
  getRiskLevelIcon(riskLevel) {
    switch (riskLevel) {
      case '안전군':
        return '🟢';
      case '주의군':
        return '🟡';
      case '고위험군':
        return '🔴';
      default:
        return '⚪';
    }
  },

  // CSV 데이터를 JSON으로 변환
  csvToJson(csvText) {
    const lines = csvText.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];

    for (let i = 1; i < lines.length; i++) {
      if (lines[i].trim()) {
        const values = lines[i].split(',').map(v => v.trim());
        const row = {};
        headers.forEach((header, index) => {
          row[header] = values[index];
        });
        data.push(row);
      }
    }

    return data;
  },

  // 파일 확장자 확인
  isValidFileType(fileName, allowedTypes = ['.csv', '.xlsx', '.json']) {
    const extension = fileName.toLowerCase().substring(fileName.lastIndexOf('.'));
    return allowedTypes.includes(extension);
  },

  // 날짜 포맷팅
  formatDate(date) {
    if (!date) return '';
    return new Date(date).toLocaleString('ko-KR');
  },

  // 성능 지표 색상 반환
  getPerformanceColor(value, metric) {
    if (typeof value !== 'number') return '#666';
    
    switch (metric) {
      case 'f1_score':
      case 'precision':
      case 'recall':
      case 'accuracy':
      case 'auc':
        if (value >= 0.8) return '#52c41a';
        if (value >= 0.6) return '#faad14';
        return '#ff4d4f';
      default:
        return '#666';
    }
  },

  // 최적화 방법 한글명 반환
  getMethodName(method) {
    const methodNames = {
      'grid': 'Grid Search',
      'bayesian': 'Bayesian Optimization',
      'scipy': 'Scipy Optimization'
    };
    return methodNames[method] || method;
  }
};
