/**
 * 네트워크 요청 관리자
 * 타임아웃, 재시도, 청크 전송 등 처리
 */

class NetworkManager {
  constructor() {
    this.defaultTimeout = 30000; // 30초
    this.maxRetries = 3;
    this.retryDelay = 1000; // 1초
    this.maxChunkSize = 5 * 1024 * 1024; // 5MB
  }

  /**
   * 지수 백오프 지연
   */
  async exponentialBackoff(attempt) {
    const delay = this.retryDelay * Math.pow(2, attempt - 1);
    await new Promise(resolve => setTimeout(resolve, delay));
  }

  /**
   * 네트워크 상태 확인
   */
  async checkNetworkStatus() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch('http://localhost:5007/health', {
        method: 'GET',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      return response.ok;
    } catch (error) {
      console.warn('네트워크 상태 확인 실패:', error.message);
      return false;
    }
  }

  /**
   * 재시도 로직이 포함된 fetch
   */
  async fetchWithRetry(url, options = {}, customRetries = null) {
    const maxAttempts = customRetries || this.maxRetries;
    let lastError;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      let timeoutId; // 스코프를 넓혀서 catch 블록에서도 접근 가능하도록
      
      try {
        console.log(`🔄 네트워크 요청 시도 ${attempt}/${maxAttempts}: ${url}`);
        
        // 타임아웃 제거 - 대용량 데이터 전송을 위해 무제한 대기
        const controller = new AbortController();
        
        // 타임아웃을 설정하지 않음 (무제한 대기)
        // 사용자가 명시적으로 타임아웃을 요청한 경우에만 설정
        if (options.forceTimeout) {
          const timeout = options.timeout || this.defaultTimeout;
          timeoutId = setTimeout(() => {
            console.warn(`⏰ 강제 타임아웃 (${timeout}ms): ${url}`);
            controller.abort();
          }, timeout);
        }
        
        const fetchOptions = {
          ...options,
          signal: controller.signal
        };
        
        // 타임아웃 ID를 저장해서 정리할 수 있도록 함
        const response = await fetch(url, fetchOptions);
        clearTimeout(timeoutId);
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        console.log(`✅ 네트워크 요청 성공 (시도 ${attempt})`);
        return response;
        
      } catch (error) {
        // 타임아웃 정리
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
        
        lastError = error;
        console.warn(`❌ 네트워크 요청 실패 (시도 ${attempt}):`, error.message);
        
        // AbortError의 경우 더 구체적인 메시지 제공
        if (error.name === 'AbortError') {
          if (options.forceTimeout) {
            console.warn(`⏰ 요청이 중단됨 - 강제 타임아웃`);
          } else {
            console.warn(`🛑 요청이 중단됨 - 사용자 취소 또는 네트워크 문제`);
          }
        }
        
        // 마지막 시도가 아니면 대기 후 재시도
        if (attempt < maxAttempts) {
          await this.exponentialBackoff(attempt);
          
          // 네트워크 상태 재확인
          const networkOk = await this.checkNetworkStatus();
          if (!networkOk) {
            console.warn('네트워크 연결 불안정 - 추가 대기');
            await new Promise(resolve => setTimeout(resolve, 2000));
          }
        }
      }
    }
    
    throw new Error(`모든 재시도 실패: ${lastError.message}`);
  }

  /**
   * 대용량 데이터 청크 전송
   */
  async sendLargeData(url, data, options = {}) {
    const dataString = JSON.stringify(data);
    const dataSize = new Blob([dataString]).size;
    
    console.log(`📊 전송할 데이터 크기: ${Math.round(dataSize/1024/1024*100)/100}MB`);
    
    // 작은 데이터는 일반 전송
    if (dataSize <= this.maxChunkSize) {
      return await this.fetchWithRetry(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: dataString,
        // timeout 제거 - 무제한 대기
        ...options
      });
    }
    
    // 대용량 데이터는 청크 전송
    return await this.sendDataInChunks(url, data, options);
  }

  /**
   * 청크 단위 데이터 전송
   */
  async sendDataInChunks(url, data, options = {}) {
    try {
      const resultArray = data.results || data.data || (Array.isArray(data) ? data : []);
      
      // 청크 크기 계산 (네트워크 안정성 고려)
      const sampleSize = JSON.stringify(resultArray[0] || {}).length;
      const optimalChunkSize = Math.max(50, Math.min(200, Math.floor(this.maxChunkSize / (sampleSize * 2))));
      
      console.log(`📦 청크 전송 시작: 총 ${resultArray.length}명, 청크당 ${optimalChunkSize}명`);
      
      const chunks = [];
      for (let i = 0; i < resultArray.length; i += optimalChunkSize) {
        chunks.push({
          chunkIndex: Math.floor(i / optimalChunkSize),
          totalChunks: Math.ceil(resultArray.length / optimalChunkSize),
          startIndex: i,
          endIndex: Math.min(i + optimalChunkSize, resultArray.length),
          data: resultArray.slice(i, i + optimalChunkSize)
        });
      }
      
      // 세션 ID 생성
      const sessionId = `chunk_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // 청크 전송 시작 알림
      await this.fetchWithRetry(`${url}/start-chunk-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: sessionId,
          totalChunks: chunks.length,
          totalEmployees: resultArray.length,
          metadata: {
            applied_settings: data.applied_settings,
            analysis_metadata: data.analysis_metadata
          }
        }),
        // timeout 제거 - 무제한 대기
      });
      
      // 각 청크 전송
      const chunkResults = [];
      for (let i = 0; i < chunks.length; i++) {
        try {
          console.log(`📤 청크 ${i + 1}/${chunks.length} 전송 중...`);
          
          const chunkResponse = await this.fetchWithRetry(`${url}/send-chunk`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              sessionId: sessionId,
              ...chunks[i]
            }),
            // timeout 제거 - 무제한 대기
          });
          
          const result = await chunkResponse.json();
          chunkResults.push(result);
          
          console.log(`✅ 청크 ${i + 1} 전송 완료`);
          
          // 청크 간 짧은 대기 (서버 부하 방지)
          if (i < chunks.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }
          
        } catch (chunkError) {
          console.error(`❌ 청크 ${i + 1} 전송 실패:`, chunkError);
          
          // 청크 전송 실패 시 세션 정리
          try {
            await this.fetchWithRetry(`${url}/cleanup-chunk-session`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ sessionId: sessionId }),
              // timeout 제거 - 무제한 대기
            });
          } catch (cleanupError) {
            console.warn('세션 정리 실패:', cleanupError);
          }
          
          throw chunkError;
        }
      }
      
      // 청크 전송 완료 알림
      const finalResponse = await this.fetchWithRetry(`${url}/complete-chunk-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: sessionId,
          chunkResults: chunkResults
        }),
        // timeout 제거 - 무제한 대기
      });
      
      console.log(`🎉 청크 전송 완료: ${chunks.length}개 청크 성공`);
      return finalResponse;
      
    } catch (error) {
      console.error('청크 전송 실패:', error);
      throw error;
    }
  }

  /**
   * 배치 분석 결과 저장 (개선된 버전)
   */
  async saveBatchAnalysisResults(results, options = {}) {
    const baseUrl = 'http://localhost:5007/api/batch-analysis/save-results';
    
    try {
      // 1. 네트워크 상태 확인
      const networkOk = await this.checkNetworkStatus();
      if (!networkOk) {
        throw new Error('Integration 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.');
      }
      
      // 2. 데이터 크기에 따른 전송 방식 선택
      const dataSize = new Blob([JSON.stringify(results)]).size;
      console.log(`📊 전송할 데이터 크기: ${Math.round(dataSize/1024/1024*100)/100}MB`);
      
      if (dataSize > this.maxChunkSize) {
        console.log('🔄 대용량 데이터 - 청크 전송 방식 사용');
        return await this.sendLargeData(baseUrl, results, options);
      } else {
        console.log('🔄 일반 크기 데이터 - 직접 전송');
        return await this.fetchWithRetry(baseUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(results),
          // timeout 제거 - 무제한 대기
          ...options
        });
      }
      
    } catch (error) {
      console.error('배치 결과 저장 실패:', error);
      
      // 상세한 오류 정보 제공
      if (error.name === 'AbortError') {
        throw new Error('요청 시간 초과: 서버 응답이 너무 오래 걸립니다.');
      } else if (error.message.includes('ERR_CONNECTION_REFUSED')) {
        throw new Error('서버 연결 거부: Integration 서버가 실행되지 않았습니다.');
      } else if (error.message.includes('ERR_CONNECTION_RESET')) {
        throw new Error('연결 재설정: 서버에서 연결을 끊었습니다. 데이터가 너무 클 수 있습니다.');
      } else {
        throw error;
      }
    }
  }

  /**
   * 계층적 구조 저장 (개선된 버전)
   */
  async saveHierarchicalResults(results, options = {}) {
    const baseUrl = 'http://localhost:5007/api/batch-analysis/save-hierarchical';
    
    try {
      console.log('🏢 계층적 구조 저장 시작...');
      
      // 네트워크 상태 확인
      const networkOk = await this.checkNetworkStatus();
      if (!networkOk) {
        console.warn('서버 연결 불가 - 로컬 저장으로 대체');
        return { success: false, fallback: true, message: '서버 연결 불가로 로컬 저장됨' };
      }
      
      // 부서별로 데이터 분할하여 전송
      const departments = this.groupByDepartment(results);
      const savePromises = [];
      
      for (const [deptName, deptData] of Object.entries(departments)) {
        const deptPromise = this.fetchWithRetry(`${baseUrl}/${encodeURIComponent(deptName)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(deptData),
          // timeout 제거 - 무제한 대기
        }).catch(error => {
          console.warn(`부서 ${deptName} 저장 실패:`, error.message);
          return { success: false, department: deptName, error: error.message };
        });
        
        savePromises.push(deptPromise);
      }
      
      const results_array = await Promise.allSettled(savePromises);
      const successful = results_array.filter(r => r.status === 'fulfilled').length;
      
      console.log(`🏢 계층적 저장 완료: ${successful}/${Object.keys(departments).length}개 부서`);
      
      return {
        success: successful > 0,
        totalDepartments: Object.keys(departments).length,
        successfulDepartments: successful,
        results: results_array
      };
      
    } catch (error) {
      console.error('계층적 저장 실패:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * 부서별 데이터 그룹화
   */
  groupByDepartment(results) {
    const departments = {};
    const resultArray = results.results || results.data || (Array.isArray(results) ? results : []);
    
    resultArray.forEach(employee => {
      const dept = employee.department || 
                   employee.analysis_result?.employee_data?.Department || 
                   'Unclassified';
      
      if (!departments[dept]) {
        departments[dept] = {
          department: dept,
          employees: [],
          statistics: {
            total: 0,
            high_risk: 0,
            medium_risk: 0,
            low_risk: 0
          }
        };
      }
      
      departments[dept].employees.push(employee);
      departments[dept].statistics.total++;
      
      // 위험도 통계 업데이트
      const score = employee.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 0;
      if (score >= 0.7) departments[dept].statistics.high_risk++;
      else if (score >= 0.3) departments[dept].statistics.medium_risk++;
      else departments[dept].statistics.low_risk++;
    });
    
    return departments;
  }

  /**
   * 연결 상태 모니터링
   */
  startConnectionMonitoring(callback) {
    const checkInterval = 30000; // 30초마다 확인
    
    const monitor = setInterval(async () => {
      const isConnected = await this.checkNetworkStatus();
      if (callback) {
        callback(isConnected);
      }
    }, checkInterval);
    
    return monitor; // 모니터링 중지용 ID 반환
  }

  /**
   * 모니터링 중지
   */
  stopConnectionMonitoring(monitorId) {
    if (monitorId) {
      clearInterval(monitorId);
    }
  }
}

const networkManager = new NetworkManager();
export default networkManager;
