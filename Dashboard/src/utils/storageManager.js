/**
 * 대용량 데이터 저장 관리자
 * LocalStorage 한계 해결 및 IndexedDB 활용
 */

class StorageManager {
  constructor() {
    this.dbName = 'AgenticAnalysisDB';
    this.dbVersion = 1;
    this.db = null;
    this.maxLocalStorageSize = 4 * 1024 * 1024; // 4MB 안전 한계
  }

  /**
   * IndexedDB 초기화
   */
  async initDB() {
    return new Promise((resolve, reject) => {
      if (this.db) {
        resolve(this.db);
        return;
      }

      const request = indexedDB.open(this.dbName, this.dbVersion);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // 배치 분석 결과 저장소
        if (!db.objectStoreNames.contains('batchResults')) {
          const batchStore = db.createObjectStore('batchResults', { keyPath: 'id' });
          batchStore.createIndex('timestamp', 'timestamp', { unique: false });
          batchStore.createIndex('department', 'department', { unique: false });
        }
        
        // 청크 데이터 저장소
        if (!db.objectStoreNames.contains('dataChunks')) {
          const chunkStore = db.createObjectStore('dataChunks', { keyPath: 'id' });
          chunkStore.createIndex('batchId', 'batchId', { unique: false });
          chunkStore.createIndex('chunkIndex', 'chunkIndex', { unique: false });
        }
        
        // 메타데이터 저장소
        if (!db.objectStoreNames.contains('metadata')) {
          db.createObjectStore('metadata', { keyPath: 'key' });
        }
      };
    });
  }

  /**
   * 데이터 크기 계산
   */
  calculateDataSize(data) {
    const jsonString = JSON.stringify(data);
    return new Blob([jsonString]).size;
  }

  /**
   * 데이터 압축 (간단한 압축)
   */
  compressData(data) {
    const jsonString = JSON.stringify(data);
    
    // 불필요한 공백 제거 및 중복 키 최적화
    const compressed = jsonString
      .replace(/\s+/g, ' ')
      .replace(/,\s*}/g, '}')
      .replace(/{\s*/g, '{')
      .replace(/\[\s*/g, '[')
      .replace(/\s*\]/g, ']');
    
    return compressed;
  }

  /**
   * 스마트 데이터 저장 (크기에 따라 자동 선택)
   */
  async saveAnalysisResults(results, options = {}) {
    const timestamp = options.timestamp || new Date().toISOString();
    const batchId = options.batchId || `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    console.log('🔄 스마트 저장 시스템 시작...');
    
    try {
      // 1. 데이터 크기 확인
      const dataSize = this.calculateDataSize(results);
      console.log(`📊 원본 데이터 크기: ${Math.round(dataSize/1024/1024*100)/100}MB`);
      
      // 2. 저장 방식 결정
      if (dataSize < this.maxLocalStorageSize) {
        // LocalStorage 사용 (빠른 접근)
        return await this.saveToLocalStorage(results, batchId, timestamp);
      } else {
        // IndexedDB 사용 (대용량)
        return await this.saveToIndexedDB(results, batchId, timestamp);
      }
    } catch (error) {
      console.error('스마트 저장 실패:', error);
      // 폴백: 요약 데이터만 저장
      return await this.saveSummaryOnly(results, batchId, timestamp);
    }
  }

  /**
   * LocalStorage 저장 (최적화된 청크 방식)
   */
  async saveToLocalStorage(results, batchId, timestamp) {
    try {
      // 기존 데이터 정리
      this.clearOldLocalStorageData();
      
      const resultArray = results.results || results.data || (Array.isArray(results) ? results : []);
      
      // 압축된 데이터 크기로 청크 크기 계산
      const sampleData = this.compressData(resultArray[0] || {});
      const sampleSize = new Blob([sampleData]).size;
      const optimalChunkSize = Math.max(50, Math.min(200, Math.floor(this.maxLocalStorageSize / (sampleSize * 2))));
      
      console.log(`📦 LocalStorage 청크 저장: 청크당 ${optimalChunkSize}명`);
      
      const chunks = [];
      for (let i = 0; i < resultArray.length; i += optimalChunkSize) {
        const chunkData = resultArray.slice(i, i + optimalChunkSize);
        const compressedChunk = this.compressData(chunkData);
        
        chunks.push({
          id: `${batchId}_chunk_${Math.floor(i / optimalChunkSize)}`,
          batchId: batchId,
          chunkIndex: Math.floor(i / optimalChunkSize),
          startIndex: i,
          endIndex: Math.min(i + optimalChunkSize, resultArray.length),
          data: compressedChunk,
          compressed: true
        });
      }
      
      // 청크 저장
      let savedChunks = 0;
      for (const chunk of chunks) {
        try {
          localStorage.setItem(`batch_chunk_${chunk.id}`, JSON.stringify(chunk));
          savedChunks++;
        } catch (chunkError) {
          console.warn(`청크 ${chunk.id} 저장 실패:`, chunkError);
          break;
        }
      }
      
      // 메타데이터 저장
      const metadata = {
        batchId: batchId,
        timestamp: timestamp,
        totalEmployees: resultArray.length,
        savedEmployees: Math.min(savedChunks * optimalChunkSize, resultArray.length),
        totalChunks: chunks.length,
        savedChunks: savedChunks,
        chunkSize: optimalChunkSize,
        storageType: 'localStorage_compressed',
        compressed: true
      };
      
      localStorage.setItem('batchAnalysisMetadata', JSON.stringify(metadata));
      
      console.log(`✅ LocalStorage 저장 완료: ${savedChunks}/${chunks.length}개 청크`);
      return { success: true, method: 'localStorage', savedChunks, totalChunks: chunks.length };
      
    } catch (error) {
      console.error('LocalStorage 저장 실패:', error);
      throw error;
    }
  }

  /**
   * IndexedDB 저장 (대용량 데이터용)
   */
  async saveToIndexedDB(results, batchId, timestamp) {
    try {
      await this.initDB();
      
      const resultArray = results.results || results.data || (Array.isArray(results) ? results : []);
      const chunkSize = 500; // IndexedDB는 더 큰 청크 사용 가능
      
      console.log(`🗄️ IndexedDB 저장: ${resultArray.length}명, 청크당 ${chunkSize}명`);
      
      const transaction = this.db.transaction(['batchResults', 'dataChunks', 'metadata'], 'readwrite');
      const batchStore = transaction.objectStore('batchResults');
      const chunkStore = transaction.objectStore('dataChunks');
      const metaStore = transaction.objectStore('metadata');
      
      // 배치 메타데이터 저장
      const batchMetadata = {
        id: batchId,
        timestamp: timestamp,
        totalEmployees: resultArray.length,
        totalChunks: Math.ceil(resultArray.length / chunkSize),
        storageType: 'indexedDB',
        compressed: false
      };
      
      await this.promisifyRequest(batchStore.put(batchMetadata));
      
      // 청크 단위로 저장
      const chunkPromises = [];
      for (let i = 0; i < resultArray.length; i += chunkSize) {
        const chunkData = {
          id: `${batchId}_chunk_${Math.floor(i / chunkSize)}`,
          batchId: batchId,
          chunkIndex: Math.floor(i / chunkSize),
          startIndex: i,
          endIndex: Math.min(i + chunkSize, resultArray.length),
          data: resultArray.slice(i, i + chunkSize),
          timestamp: timestamp
        };
        
        chunkPromises.push(this.promisifyRequest(chunkStore.put(chunkData)));
      }
      
      await Promise.all(chunkPromises);
      
      // 전역 메타데이터 업데이트
      await this.promisifyRequest(metaStore.put({
        key: 'lastBatchAnalysis',
        batchId: batchId,
        timestamp: timestamp,
        method: 'indexedDB'
      }));
      
      console.log(`✅ IndexedDB 저장 완료: ${chunkPromises.length}개 청크`);
      return { success: true, method: 'indexedDB', savedChunks: chunkPromises.length, totalChunks: chunkPromises.length };
      
    } catch (error) {
      console.error('IndexedDB 저장 실패:', error);
      throw error;
    }
  }

  /**
   * 요약 데이터만 저장 (최후의 수단)
   */
  async saveSummaryOnly(results, batchId, timestamp) {
    try {
      localStorage.clear();
      
      const resultArray = results.results || results.data || (Array.isArray(results) ? results : []);
      
      // 위험도별 통계만 저장
      const summary = {
        batchId: batchId,
        timestamp: timestamp,
        totalEmployees: resultArray.length,
        storageType: 'summary_only',
        riskDistribution: {
          high: resultArray.filter(r => {
            const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
            return score && score >= 0.7;
          }).length,
          medium: resultArray.filter(r => {
            const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
            return score && score >= 0.3 && score < 0.7;
          }).length,
          low: resultArray.filter(r => {
            const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
            return score && score < 0.3;
          }).length
        },
        departmentStats: this.calculateDepartmentStats(resultArray)
      };
      
      localStorage.setItem('batchAnalysisSummary', JSON.stringify(summary));
      
      console.log('⚠️ 요약 데이터만 저장됨 (용량 제한으로 인해)');
      return { success: true, method: 'summary', warning: '용량 제한으로 요약 데이터만 저장됨' };
      
    } catch (error) {
      console.error('요약 저장도 실패:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * 부서별 통계 계산
   */
  calculateDepartmentStats(resultArray) {
    const deptStats = {};
    
    resultArray.forEach(emp => {
      const dept = emp.department || emp.analysis_result?.employee_data?.Department || 'Unknown';
      const score = emp.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 0;
      
      if (!deptStats[dept]) {
        deptStats[dept] = { total: 0, high: 0, medium: 0, low: 0 };
      }
      
      deptStats[dept].total++;
      if (score >= 0.7) deptStats[dept].high++;
      else if (score >= 0.3) deptStats[dept].medium++;
      else deptStats[dept].low++;
    });
    
    return deptStats;
  }

  /**
   * 이전 LocalStorage 데이터 정리
   */
  clearOldLocalStorageData() {
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && (key.startsWith('batch_chunk_') || key.startsWith('batchAnalysisResults'))) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(key => localStorage.removeItem(key));
  }

  /**
   * IndexedDB 요청을 Promise로 변환
   */
  promisifyRequest(request) {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * 저장된 데이터 로드
   */
  async loadAnalysisResults(batchId = null) {
    try {
      // 1. 메타데이터 확인
      const metadata = JSON.parse(localStorage.getItem('batchAnalysisMetadata') || '{}');
      
      if (metadata.storageType === 'localStorage_compressed') {
        return await this.loadFromLocalStorage(metadata);
      } else if (metadata.storageType === 'indexedDB') {
        return await this.loadFromIndexedDB(batchId || metadata.batchId);
      } else {
        // 요약 데이터만 있는 경우
        const summary = JSON.parse(localStorage.getItem('batchAnalysisSummary') || '{}');
        return { success: true, data: summary, type: 'summary' };
      }
    } catch (error) {
      console.error('데이터 로드 실패:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * LocalStorage에서 데이터 로드
   */
  async loadFromLocalStorage(metadata) {
    try {
      const results = [];
      
      for (let i = 0; i < metadata.savedChunks; i++) {
        const chunkKey = `batch_chunk_${metadata.batchId}_chunk_${i}`;
        const chunkData = localStorage.getItem(chunkKey);
        
        if (chunkData) {
          const chunk = JSON.parse(chunkData);
          const decompressedData = chunk.compressed ? JSON.parse(chunk.data) : chunk.data;
          results.push(...decompressedData);
        }
      }
      
      return { success: true, data: results, metadata: metadata };
    } catch (error) {
      console.error('LocalStorage 로드 실패:', error);
      throw error;
    }
  }

  /**
   * IndexedDB에서 데이터 로드
   */
  async loadFromIndexedDB(batchId) {
    try {
      await this.initDB();
      
      const transaction = this.db.transaction(['dataChunks'], 'readonly');
      const chunkStore = transaction.objectStore('dataChunks');
      const index = chunkStore.index('batchId');
      
      const chunks = await this.promisifyRequest(index.getAll(batchId));
      
      // 청크 순서대로 정렬
      chunks.sort((a, b) => a.chunkIndex - b.chunkIndex);
      
      const results = [];
      chunks.forEach(chunk => {
        results.push(...chunk.data);
      });
      
      return { success: true, data: results, metadata: { storageType: 'indexedDB', totalEmployees: results.length } };
    } catch (error) {
      console.error('IndexedDB 로드 실패:', error);
      throw error;
    }
  }
}

export default new StorageManager();
