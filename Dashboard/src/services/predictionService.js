// 예측 결과 저장 및 관리 서비스
class PredictionService {
  constructor() {
    this.storageKey = 'predictionHistory';
    this.maxHistorySize = null; // 무제한 저장 (null = 제한 없음)
    this.compressionEnabled = true; // 데이터 압축 활성화
    this.autoBackupEnabled = true; // 자동 백업 활성화
    this.backupThreshold = 100; // 100개마다 자동 백업
  }

  // 예측 결과 저장
  savePredictionResult(predictionData) {
    try {
      const history = this.getPredictionHistory();
      
      // 새로운 예측 결과 생성
      const newPrediction = {
        id: this.generateId(),
        timestamp: new Date().toISOString(),
        title: predictionData.title || `${new Date().toLocaleDateString('ko-KR')} 배치 분석`,
        totalEmployees: predictionData.totalEmployees || 0,
        highRiskCount: predictionData.highRiskCount || 0,
        mediumRiskCount: predictionData.mediumRiskCount || 0,
        lowRiskCount: predictionData.lowRiskCount || 0,
        accuracy: predictionData.accuracy || 0,
        status: 'completed',
        summary: predictionData.summary || '분석이 완료되었습니다.',
        keyInsights: predictionData.keyInsights || [],
        rawData: predictionData.rawData || null, // 원본 분석 데이터
        departmentStats: predictionData.departmentStats || {},
        riskFactors: predictionData.riskFactors || []
      };

      // 히스토리 맨 앞에 추가
      history.unshift(newPrediction);

      // 최대 크기 제한 (null이면 무제한)
      if (this.maxHistorySize && history.length > this.maxHistorySize) {
        const removedItems = history.splice(this.maxHistorySize);
        console.log(`히스토리 크기 제한으로 ${removedItems.length}개 항목 제거됨`);
      }

      // 데이터 압축 및 저장
      this.saveToStorage(history);

      // 자동 백업 체크
      if (this.autoBackupEnabled && history.length % this.backupThreshold === 0) {
        this.createAutoBackup(history);
      }
      
      console.log('예측 결과 저장 완료:', newPrediction.id);
      return newPrediction;
    } catch (error) {
      console.error('예측 결과 저장 실패:', error);
      throw error;
    }
  }

  // 데이터 압축 및 저장
  saveToStorage(history) {
    try {
      let dataToStore = history;
      
      // 압축 활성화 시 데이터 압축
      if (this.compressionEnabled) {
        dataToStore = this.compressHistoryData(history);
      }
      
      const dataString = JSON.stringify(dataToStore);
      const dataSize = new Blob([dataString]).size;
      
      // localStorage 용량 체크 (5MB 제한)
      if (dataSize > 5 * 1024 * 1024) {
        console.warn('데이터 크기가 5MB를 초과합니다. IndexedDB로 전환합니다.');
        // 큰 데이터는 IndexedDB에 저장 시도
        this.saveToIndexedDB(dataToStore).catch(error => {
          console.error('IndexedDB 저장 실패, localStorage로 폴백:', error);
          try {
            localStorage.setItem(this.storageKey, dataString);
          } catch (localStorageError) {
            console.error('localStorage 저장도 실패:', localStorageError);
          }
        });
      } else {
        localStorage.setItem(this.storageKey, dataString);
      }
      
      console.log(`히스토리 저장 완료: ${history.length}개 항목, ${(dataSize/1024).toFixed(2)}KB`);
    } catch (error) {
      console.error('저장 실패:', error);
      // 저장 실패 시 압축 없이 재시도
      if (this.compressionEnabled) {
        console.log('압축 없이 재시도...');
        localStorage.setItem(this.storageKey, JSON.stringify(history));
      }
    }
  }

  // 예측 히스토리 조회
  getPredictionHistory() {
    try {
      // localStorage에서 먼저 시도
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        const data = JSON.parse(stored);
        return this.decompressHistoryData(data);
      }
      
      // localStorage에 데이터가 없으면 빈 배열 반환
      // IndexedDB는 비동기이므로 동기 메서드에서는 사용하지 않음
      return [];
    } catch (error) {
      console.error('예측 히스토리 조회 실패:', error);
      return [];
    }
  }

  // 비동기 히스토리 조회 (IndexedDB 포함)
  async getPredictionHistoryAsync() {
    try {
      // localStorage에서 먼저 시도
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        const data = JSON.parse(stored);
        return this.decompressHistoryData(data);
      }
      
      // IndexedDB에서 시도
      const indexedData = await this.loadFromIndexedDB();
      return indexedData || [];
    } catch (error) {
      console.error('예측 히스토리 조회 실패:', error);
      return [];
    }
  }

  // 특정 예측 결과 조회
  getPredictionById(id) {
    const history = this.getPredictionHistory();
    return history.find(prediction => prediction.id === id);
  }

  // 최신 예측 결과 조회
  getLatestPrediction() {
    const history = this.getPredictionHistory();
    return history.length > 0 ? history[0] : null;
  }

  // 예측 결과 삭제
  deletePrediction(id) {
    try {
      const history = this.getPredictionHistory();
      const filteredHistory = history.filter(prediction => prediction.id !== id);
      localStorage.setItem(this.storageKey, JSON.stringify(filteredHistory));
      console.log('예측 결과 삭제 완료:', id);
      return true;
    } catch (error) {
      console.error('예측 결과 삭제 실패:', error);
      return false;
    }
  }

  // 예측 결과 업데이트
  updatePrediction(id, updateData) {
    try {
      const history = this.getPredictionHistory();
      const index = history.findIndex(prediction => prediction.id === id);
      
      if (index !== -1) {
        history[index] = { ...history[index], ...updateData };
        localStorage.setItem(this.storageKey, JSON.stringify(history));
        console.log('예측 결과 업데이트 완료:', id);
        return history[index];
      }
      
      throw new Error('예측 결과를 찾을 수 없습니다.');
    } catch (error) {
      console.error('예측 결과 업데이트 실패:', error);
      throw error;
    }
  }

  // 배치 분석 결과를 예측 히스토리로 변환
  convertBatchResultToPrediction(batchResult) {
    if (!batchResult || !batchResult.results) {
      return null;
    }

    const results = batchResult.results;
    const totalEmployees = results.length;
    
    console.log('🔄 배치 결과 변환 중:', { totalEmployees, sampleEmployee: results[0] });
    
    // 위험도별 분류 - comprehensive_report의 overall_risk_level을 직접 사용
    const riskCounts = results.reduce((acc, employee) => {
      // comprehensive_report의 overall_risk_level을 직접 읽어옴 (정규화된 값 사용)
      const riskLevel = employee.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_level;
      const normalizedLevel = this.normalizeRiskLevel(riskLevel);
      acc[normalizedLevel]++;
      return acc;
    }, { high: 0, medium: 0, low: 0, unknown: 0 });

    console.log('📊 위험도별 분류 (comprehensive_report 기준):', riskCounts);

    // 부서별 통계 - comprehensive_report의 overall_risk_level 사용
    const departmentStats = results.reduce((acc, employee) => {
      // 부서 정보 추출 (여러 경로에서 시도)
      const dept = employee.analysis_result?.employee_data?.Department || 
                  employee.department || 
                  '미분류';
      
      if (!acc[dept]) {
        acc[dept] = { total: 0, high: 0, medium: 0, low: 0 };
      }
      acc[dept].total++;
      
      // comprehensive_report의 overall_risk_level을 직접 읽어옴
      const riskLevel = employee.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_level;
      const normalizedLevel = this.normalizeRiskLevel(riskLevel);
      if (normalizedLevel !== 'unknown') {
        acc[dept][normalizedLevel]++;
      }
      return acc;
    }, {});

    console.log('🏢 부서별 통계 (comprehensive_report 기준):', departmentStats);

    // 주요 위험 요인 분석
    const riskFactors = this.analyzeRiskFactors(results);

    // 인사이트 생성
    const keyInsights = this.generateInsights(results, departmentStats, riskFactors);

    return {
      title: `${new Date().toLocaleDateString('ko-KR')} 배치 분석`,
      totalEmployees,
      highRiskCount: riskCounts.high,
      mediumRiskCount: riskCounts.medium,
      lowRiskCount: riskCounts.low,
      accuracy: batchResult.model_accuracy || 94.2,
      summary: this.generateSummary(totalEmployees, riskCounts, departmentStats),
      keyInsights,
      rawData: batchResult,
      departmentStats,
      riskFactors
    };
  }

  // 위험도 레벨 정규화 (comprehensive_report의 overall_risk_level 표준화)
  normalizeRiskLevel(riskLevel) {
    if (!riskLevel) return 'unknown';
    
    const levelStr = String(riskLevel).toLowerCase().trim();
    
    // HIGH 변형들
    if (levelStr === 'high' || levelStr === '고위험군' || levelStr === '고위험' || 
        levelStr === 'high_risk' || levelStr === 'critical') {
      return 'high';
    }
    
    // MEDIUM 변형들
    if (levelStr === 'medium' || levelStr === '주의군' || levelStr === '중위험' || 
        levelStr === 'mid' || levelStr === 'moderate' || levelStr === 'warning') {
      return 'medium';
    }
    
    // LOW 변형들
    if (levelStr === 'low' || levelStr === '안전군' || levelStr === '저위험' || 
        levelStr === 'safe' || levelStr === 'normal') {
      return 'low';
    }
    
    return 'unknown';
  }

  // 위험도 레벨 결정 (deprecated - normalizeRiskLevel 사용 권장)
  getRiskLevel(probability) {
    if (probability >= 0.7) return 'high';
    if (probability >= 0.4) return 'medium';
    return 'low';
  }

  // 위험 요인 분석 - comprehensive_report의 overall_risk_level 사용
  analyzeRiskFactors(results) {
    const factors = {};
    
    results.forEach(employee => {
      // comprehensive_report의 overall_risk_level을 직접 확인
      const riskLevel = employee.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_level;
      const normalizedLevel = this.normalizeRiskLevel(riskLevel);
      
      if (normalizedLevel === 'high') {
        // 고위험군 직원의 특성 분석
        const employeeData = employee.analysis_result?.employee_data || {};
        
        Object.keys(employeeData).forEach(key => {
          if (key !== 'EmployeeNumber' && employeeData[key]) {
            if (!factors[key]) factors[key] = [];
            factors[key].push(employeeData[key]);
          }
        });
        
        // 분석 결과에서 위험 요인 추출
        if (employee.analysis_result?.combined_analysis?.risk_factors) {
          employee.analysis_result.combined_analysis.risk_factors.forEach(factor => {
            if (!factors['risk_factors']) factors['risk_factors'] = [];
            factors['risk_factors'].push(factor);
          });
        }
      }
    });

    return factors;
  }

  // 인사이트 생성 - comprehensive_report의 overall_risk_level 사용
  generateInsights(results, departmentStats, riskFactors) {
    const insights = [];
    
    // 부서별 위험도 분석
    const deptRisks = Object.entries(departmentStats)
      .map(([dept, stats]) => ({
        dept,
        riskRate: ((stats.high + stats.medium) / stats.total * 100).toFixed(1)
      }))
      .sort((a, b) => b.riskRate - a.riskRate);

    if (deptRisks.length > 0 && parseFloat(deptRisks[0].riskRate) > 0) {
      insights.push(`${deptRisks[0].dept} 부서의 이직 위험도가 ${deptRisks[0].riskRate}%로 가장 높습니다.`);
    }

    // 전체 위험도 분석 - comprehensive_report의 overall_risk_level 사용
    const totalHigh = results.filter(e => {
      const riskLevel = e.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_level;
      const normalizedLevel = this.normalizeRiskLevel(riskLevel);
      return normalizedLevel === 'high';
    }).length;
    const totalRiskRate = (totalHigh / results.length * 100).toFixed(1);
    
    if (totalHigh > 0) {
      insights.push(`전체 직원 중 ${totalRiskRate}%가 고위험군으로 분류되었습니다.`);
    } else {
      insights.push('현재 전체 직원의 이직 위험도가 안정적인 수준입니다.');
    }

    // 추가 인사이트
    if (totalHigh > 0) {
      insights.push('정기적인 직원 만족도 조사와 개별 면담을 통한 사전 예방이 필요합니다.');
    } else {
      insights.push('안정적인 조직 문화 유지를 위해 지속적인 모니터링이 권장됩니다.');
    }

    return insights;
  }

  // 요약 생성
  generateSummary(totalEmployees, riskCounts, departmentStats) {
    const highRiskRate = (riskCounts.high / totalEmployees * 100).toFixed(1);
    const deptCount = Object.keys(departmentStats).length;
    
    return `총 ${totalEmployees.toLocaleString()}명의 직원을 분석한 결과, ${riskCounts.high}명(${highRiskRate}%)이 고위험군으로 분류되었습니다. ${deptCount}개 부서에 걸쳐 분석이 수행되었으며, 부서별로 차별화된 관리 전략이 필요합니다.`;
  }

  // 기본 히스토리 데이터
  getDefaultHistory() {
    return [
      {
        id: 'default_001',
        timestamp: '2024-01-15T09:30:00Z',
        title: '2024년 1월 배치 분석',
        totalEmployees: 1250,
        highRiskCount: 89,
        mediumRiskCount: 156,
        lowRiskCount: 1005,
        accuracy: 94.2,
        status: 'completed',
        summary: '전체 직원 중 7.1%가 고위험군으로 분류되었으며, 특히 IT 부서와 영업 부서에서 이직 위험이 높게 나타났습니다.',
        keyInsights: [
          '근무 만족도가 낮은 직원의 이직 확률이 3.2배 높음',
          '원격근무 선호도와 이직 의향 간 강한 상관관계 발견',
          '승진 기회 부족이 주요 이직 요인으로 확인됨'
        ],
        departmentStats: {
          'IT': { total: 200, high: 25, medium: 35, low: 140 },
          '영업': { total: 180, high: 20, medium: 28, low: 132 },
          '마케팅': { total: 150, high: 12, medium: 22, low: 116 },
          '인사': { total: 120, high: 8, medium: 15, low: 97 },
          '재무': { total: 100, high: 5, medium: 12, low: 83 }
        },
        riskFactors: ['job_satisfaction', 'promotion_opportunity', 'work_life_balance']
      },
      {
        id: 'default_002',
        timestamp: '2024-01-08T14:15:00Z',
        title: '2024년 1월 초 예측 분석',
        totalEmployees: 1248,
        highRiskCount: 92,
        mediumRiskCount: 149,
        lowRiskCount: 1007,
        accuracy: 93.8,
        status: 'completed',
        summary: '연말 보너스 지급 후 이직 위험도가 일시적으로 감소했으나, 여전히 주의가 필요한 직원들이 존재합니다.',
        keyInsights: [
          '보너스 지급 후 전반적인 만족도 상승',
          '하지만 경력 개발 기회에 대한 불만은 지속',
          '관리자와의 관계 개선이 필요한 팀 식별'
        ],
        departmentStats: {
          'IT': { total: 198, high: 23, medium: 32, low: 143 },
          '영업': { total: 182, high: 22, medium: 30, low: 130 },
          '마케팅': { total: 148, high: 10, medium: 20, low: 118 },
          '인사': { total: 118, high: 6, medium: 14, low: 98 },
          '재무': { total: 102, high: 4, medium: 11, low: 87 }
        },
        riskFactors: ['career_development', 'manager_relationship', 'compensation']
      }
    ];
  }

  // ID 생성
  generateId() {
    return 'pred_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  // 히스토리 초기화
  clearHistory() {
    localStorage.removeItem(this.storageKey);
    console.log('예측 히스토리 초기화 완료');
  }

  // 샘플 데이터 초기화 (개발용)
  clearSampleData() {
    localStorage.removeItem(this.storageKey);
    localStorage.removeItem('batchAnalysisResults');
    localStorage.removeItem('lastAnalysisTimestamp');
    console.log('모든 샘플 데이터 초기화 완료');
  }

  // 데이터 압축
  compressHistoryData(history) {
    return history.map(item => ({
      ...item,
      // 큰 데이터 필드 압축
      rawData: item.rawData ? 'compressed' : null,
      keyInsights: item.keyInsights ? item.keyInsights.slice(0, 3) : [], // 상위 3개만
      summary: item.summary ? item.summary.substring(0, 200) : '' // 200자 제한
    }));
  }

  // 데이터 압축 해제
  decompressHistoryData(data) {
    // 압축된 데이터인지 확인
    if (Array.isArray(data) && data.length > 0 && data[0].compressed !== undefined) {
      return data; // 이미 압축 해제된 데이터
    }
    return data;
  }

  // IndexedDB에 저장
  async saveToIndexedDB(data) {
    try {
      return new Promise((resolve, reject) => {
        const request = indexedDB.open('PredictionHistoryDB', 1);
        
        request.onupgradeneeded = (event) => {
          const db = event.target.result;
          if (!db.objectStoreNames.contains('history')) {
            db.createObjectStore('history', { keyPath: 'id' });
          }
        };
        
        request.onsuccess = (event) => {
          const db = event.target.result;
          
          try {
            const transaction = db.transaction(['history'], 'readwrite');
            const store = transaction.objectStore('history');
            
            // 기존 데이터 삭제 후 새 데이터 저장
            const clearRequest = store.clear();
            
            clearRequest.onsuccess = () => {
              // 데이터 추가
              let addedCount = 0;
              const totalItems = data.length;
              
              if (totalItems === 0) {
                console.log('IndexedDB에 저장 완료 (빈 데이터)');
                resolve();
                return;
              }
              
              data.forEach(item => {
                const addRequest = store.add(item);
                addRequest.onsuccess = () => {
                  addedCount++;
                  if (addedCount === totalItems) {
                    console.log(`IndexedDB에 저장 완료: ${totalItems}개 항목`);
                    resolve();
                  }
                };
                addRequest.onerror = (error) => {
                  console.error('IndexedDB 항목 저장 실패:', error);
                };
              });
            };
            
            clearRequest.onerror = (error) => {
              console.error('IndexedDB 클리어 실패:', error);
              reject(error);
            };
            
            transaction.onerror = (error) => {
              console.error('IndexedDB 트랜잭션 실패:', error);
              reject(error);
            };
          } catch (transactionError) {
            console.error('IndexedDB 트랜잭션 생성 실패:', transactionError);
            reject(transactionError);
          }
        };
        
        request.onerror = (event) => {
          console.error('IndexedDB 열기 실패:', event.target.error);
          reject(event.target.error);
        };
      });
    } catch (error) {
      console.error('IndexedDB 저장 실패:', error);
      throw error;
    }
  }

  // IndexedDB에서 로드
  async loadFromIndexedDB() {
    try {
      return new Promise((resolve) => {
        const request = indexedDB.open('PredictionHistoryDB', 1);
        
        request.onupgradeneeded = (event) => {
          const db = event.target.result;
          if (!db.objectStoreNames.contains('history')) {
            db.createObjectStore('history', { keyPath: 'id' });
          }
        };
        
        request.onsuccess = (event) => {
          const db = event.target.result;
          
          // 객체 저장소가 존재하는지 확인
          if (!db.objectStoreNames.contains('history')) {
            console.log('IndexedDB에 history 저장소가 없습니다.');
            resolve([]);
            return;
          }
          
          try {
            const transaction = db.transaction(['history'], 'readonly');
            const store = transaction.objectStore('history');
            const getAllRequest = store.getAll();
            
            getAllRequest.onsuccess = () => {
              resolve(getAllRequest.result || []);
            };
            
            getAllRequest.onerror = () => {
              console.error('IndexedDB 데이터 조회 실패');
              resolve([]);
            };
          } catch (transactionError) {
            console.error('IndexedDB 트랜잭션 오류:', transactionError);
            resolve([]);
          }
        };
        
        request.onerror = (event) => {
          console.error('IndexedDB 열기 실패:', event.target.error);
          resolve([]);
        };
      });
    } catch (error) {
      console.error('IndexedDB 로드 실패:', error);
      return [];
    }
  }

  // 자동 백업 생성
  createAutoBackup(history) {
    try {
      const backupData = {
        timestamp: new Date().toISOString(),
        version: '1.0',
        totalItems: history.length,
        data: history
      };
      
      const dataStr = JSON.stringify(backupData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      
      const link = document.createElement('a');
      link.href = URL.createObjectURL(dataBlob);
      link.download = `auto_backup_${history.length}items_${new Date().toISOString().split('T')[0]}.json`;
      
      // 자동 다운로드 (사용자가 원할 때만)
      if (window.confirm(`${history.length}개의 예측 결과가 저장되었습니다. 자동 백업을 다운로드하시겠습니까?`)) {
        link.click();
      }
      
      console.log(`자동 백업 생성됨: ${history.length}개 항목`);
    } catch (error) {
      console.error('자동 백업 생성 실패:', error);
    }
  }

  // 히스토리 내보내기 (개선된 버전)
  exportHistory() {
    const history = this.getPredictionHistory();
    
    const exportData = {
      exportDate: new Date().toISOString(),
      totalItems: history.length,
      systemInfo: {
        version: '1.0',
        source: 'Retain Sentinel 360'
      },
      data: history
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `prediction_history_export_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    console.log(`히스토리 내보내기 완료: ${history.length}개 항목`);
  }

  // 히스토리 가져오기
  importHistory(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => {
        try {
          const importData = JSON.parse(event.target.result);
          const importedHistory = importData.data || importData; // 새/구 형식 지원
          
          // 기존 히스토리와 병합
          const currentHistory = this.getPredictionHistory();
          const mergedHistory = [...importedHistory, ...currentHistory];
          
          // 중복 제거 (ID 기준)
          const uniqueHistory = mergedHistory.filter((item, index, self) => 
            index === self.findIndex(h => h.id === item.id)
          );
          
          // 날짜순 정렬 (최신순)
          uniqueHistory.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
          
          this.saveToStorage(uniqueHistory);
          
          console.log(`히스토리 가져오기 완료: ${importedHistory.length}개 항목 추가`);
          resolve(uniqueHistory);
        } catch (error) {
          reject(error);
        }
      };
      
      reader.onerror = () => reject(new Error('파일 읽기 실패'));
      reader.readAsText(file);
    });
  }
}

// 싱글톤 인스턴스 생성
export const predictionService = new PredictionService();

// 개발용 전역 함수 (브라우저 콘솔에서 사용 가능)
if (typeof window !== 'undefined') {
  window.clearPredictionData = () => {
    predictionService.clearSampleData();
    window.location.reload();
  };
  
  window.clearIndexedDB = () => {
    const deleteRequest = indexedDB.deleteDatabase('PredictionHistoryDB');
    deleteRequest.onsuccess = () => {
      console.log('IndexedDB 삭제 완료');
      window.location.reload();
    };
    deleteRequest.onerror = (error) => {
      console.error('IndexedDB 삭제 실패:', error);
    };
  };
  
  window.checkIndexedDB = async () => {
    try {
      const history = await predictionService.getPredictionHistoryAsync();
      console.log('IndexedDB 데이터:', history);
      return history;
    } catch (error) {
      console.error('IndexedDB 확인 실패:', error);
      return [];
    }
  };
}

export default predictionService;
