import React, { useState, useEffect, useRef } from 'react';
import { notification } from 'antd';
import {
  HomeOutlined,
  ApiOutlined,
  RobotOutlined,
  TeamOutlined,
  FileTextOutlined,
  BulbOutlined,
  WarningOutlined,
  AimOutlined,
  SettingOutlined,
  ExperimentOutlined,
} from '@ant-design/icons';

import Home from './components/Home';
import BatchAnalysis from './components/BatchAnalysis';
import PostAnalysis from './components/PostAnalysis';
import ReportGeneration from './components/ReportGeneration';
import RelationshipAnalysis from './components/RelationshipAnalysis';
import GroupStatistics from './components/GroupStatistics';
import Login, { getStoredUser, logout, PWC_LOGO } from './components/Login';
import { apiService } from './services/apiService';
import './styles/typography.css'; // 통일된 폰트 크기 체계

// ── 새 탭 Placeholder 컴포넌트들 ──
const PlaceholderPage = ({ title, icon, description, items }) => (
  <div style={{ padding: '0 8px' }}>
    <div style={{
      background: 'linear-gradient(135deg, #2d2d2d, #4a4a4a)',
      borderRadius: 12, padding: '24px', color: '#fff', marginBottom: 20,
    }}>
      <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>{icon} {title}</div>
      <div style={{ fontSize: 13, color: '#ccc' }}>{description}</div>
    </div>
    {items && items.map((item, i) => (
      <div key={i} style={{
        background: '#fff', borderRadius: 12, padding: 20, marginBottom: 16,
        borderLeft: `4px solid ${item.color || '#d93954'}`,
        boxShadow: '0 1px 4px rgba(0,0,0,.06)',
      }}>
        <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 8 }}>{item.title}</div>
        <div style={{ fontSize: 13, color: '#555', lineHeight: 1.7 }}>{item.desc}</div>
        {item.chips && (
          <div style={{ marginTop: 10, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {item.chips.map((c, j) => (
              <span key={j} style={{
                padding: '3px 8px', borderRadius: 4, fontSize: 10, fontWeight: 600,
                background: c.bg, color: c.color,
              }}>{c.label}</span>
            ))}
          </div>
        )}
      </div>
    ))}
  </div>
);

const agentChips = {
  str: { label: 'Structura', bg: '#fde8ec', color: '#d93954' },
  cog: { label: 'Cognita', bg: '#e8f0fe', color: '#2563eb' },
  chr: { label: 'Chronos', bg: '#fef3e2', color: '#e8721a' },
  sen: { label: 'Sentio', bg: '#f3e8fd', color: '#7c3aed' },
  ago: { label: 'Agora', bg: '#e6f6ec', color: '#2ea44f' },
};

const InsightsPlaceholder = () => (
  <PlaceholderPage
    title="AI 핵심 인사이트" icon="💡"
    description="5개 전문 Worker Agent의 분석 결과를 종합하여 360도 관점의 퇴사 위험 진단 제공"
    items={[
      { title: '🚨 번아웃 직전 그룹의 급속 확산', color: '#d93954',
        desc: '고위험군 중 상당수가 번아웃 직전 상태. 평균 초과근무시간 전사 대비 2.3배, 직무만족도 하위 15%.',
        chips: [agentChips.str, agentChips.sen, agentChips.chr] },
      { title: '🔗 관계망 단절 패턴 감지', color: '#2563eb',
        desc: '최근 6개월간 신규 협업 관계 미형성 직원이 고위험군에서 72%. 사회적 고립 지수 지속 상승.',
        chips: [agentChips.cog, agentChips.chr] },
      { title: '📈 행동 패턴 이상 징후 증가', color: '#e8721a',
        desc: '최근 3주간 로그인 시간 불규칙성 전분기 대비 38% 증가. 이직 준비 행동 패턴 관찰.',
        chips: [agentChips.chr, agentChips.ago] },
      { title: '💬 감성 분석: 부정 감정 키워드 급증', color: '#7c3aed',
        desc: '코칭 면담 및 자기평가 텍스트에서 "소진", "불확실", "답답함" 등 부정 키워드 45% 증가.',
        chips: [agentChips.sen] },
      { title: '🎯 외부 시장 Pull Factor 강화', color: '#2ea44f',
        desc: 'Technology, R&D 부서 LinkedIn 접속 빈도 전분기 대비 62% 증가. 시장 보상 15~25% 높음.',
        chips: [agentChips.ago, agentChips.str] },
    ]}
  />
);

const RiskFactorsPlaceholder = () => (
  <PlaceholderPage
    title="퇴사 위험 요인 분석" icon="⚠️"
    description="SHAP 기반 위험 요인 분석 및 Persona별 위험 프로필"
    items={[
      { title: '📊 전사 Top 위험 요인 (SHAP 기반)', color: '#d93954',
        desc: '1. 초과근무시간 (0.82)\n2. 직무만족도 (0.74)\n3. 연봉 동료대비 (0.68)\n4. 승진 후 경과기간 (0.62)\n5. 사회적 고립지수 (0.58)' },
      { title: '🧩 Persona별 위험 프로필', color: '#e8721a',
        desc: 'P01 번아웃 직전 — 초과근무 78%, 직무만족도 45%\nP02 보상 실망 — 보상 격차 72%, 외부 비교 65%\nP03 성장 정체 — 승진 경과 68%, PM 경험 부족 55%\nP04 보상체감 낮음 — 인센티브 미수령 58%' },
    ]}
  />
);

const InterventionPlaceholder = () => (
  <PlaceholderPage
    title="맞춤 개입 전략 프레임워크" icon="🎯"
    description="각 Agent의 분석 결과를 기반으로 Persona별 맞춤형 개입 전략을 제시합니다."
    items={[
      { title: '구조적 불만족 (Structura)', color: '#d93954',
        desc: '• 총체적 보상 검토 및 시장 대비 경쟁력 확보\n• 투명한 경력 경로 제시 및 승진 기준 명확화\n• 업무량 분배 재검토 및 초과근무 관리',
        chips: [agentChips.str] },
      { title: '관계적 단절 (Cognita)', color: '#2563eb',
        desc: '• 관리자 주도 1:1 Communication 체계 강화\n• 프로젝트 페어링을 통한 협업 기회 확대\n• 멘토링 프로그램 연계 및 네트워크 복원',
        chips: [agentChips.cog] },
      { title: '행동적 이탈 (Chronos)', color: '#e8721a',
        desc: '• 자율성 부여 및 유의미한 업무 재할당\n• 업무량 및 기대 수준 재조정\n• 시의적절한 인정과 격려 제공',
        chips: [agentChips.chr] },
      { title: '심리적 소진 (Sentio)', color: '#7c3aed',
        desc: '• JD-R 모델 기반 관리적 개입\n• Job Crafting 기법 도입\n• 웰니스 프로그램 지원 및 EAP 연계',
        chips: [agentChips.sen] },
      { title: '외부 시장 요인 (Agora)', color: '#2ea44f',
        desc: '• 내부 탤런트 마켓플레이스 관점의 관리\n• 전략적 보상 조정 (시장 벤치마크 기반)\n• 경쟁 우위 EVP 강화',
        chips: [agentChips.ago] },
    ]}
  />
);

const AdminSettingsPlaceholder = () => (
  <div style={{ padding: '0 8px' }}>
    <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#d93954', display: 'inline-block' }} />
      시스템 설정
    </div>
    {[
      { title: '위험 임계값 설정', desc: '고위험 임계값: 0.70 | 잠재적 위험 임계값: 0.40' },
      { title: 'Agent별 가중치', desc: 'Structura: 30% | Cognita: 20% | Chronos: 20% | Sentio: 15% | Agora: 15%' },
      { title: '알림 설정', desc: '고위험 신규 감지 알림: ON | 위험 등급 변동 알림: ON | 주간 요약 리포트: ON' },
      { title: '데이터 수집 주기', desc: 'HRIS (Structura): 매주 | 관계망 (Cognita): 매주 | 행동 로그 (Chronos): 매일 | 텍스트 (Sentio): 매월 | 외부 시장 (Agora): 매주' },
    ].map((item, i) => (
      <div key={i} style={{
        background: '#fff', borderRadius: 12, padding: 16, marginBottom: 12,
        border: '1px solid #eee', boxShadow: '0 1px 4px rgba(0,0,0,.06)',
      }}>
        <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6 }}>⚙️ {item.title}</div>
        <div style={{ fontSize: 13, color: '#555' }}>{item.desc}</div>
      </div>
    ))}
  </div>
);

// IndexedDB 헬퍼 함수들
  const initializeIndexedDB = () => {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('AgenticAnalysisDB', 2);
      
      request.onerror = () => {
        console.error('IndexedDB 초기화 실패:', request.error);
        reject(request.error);
      };
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        const oldVersion = event.oldVersion;
        const newVersion = event.newVersion;
        
        console.log(`IndexedDB 스키마 업그레이드: ${oldVersion} → ${newVersion}`);
        
        // 기존 object store 삭제 후 재생성 (안전한 업그레이드)
        if (db.objectStoreNames.contains('results')) {
          db.deleteObjectStore('results');
          console.log('기존 IndexedDB object store "results" 삭제');
        }
        
        db.createObjectStore('results', { keyPath: 'id' });
        console.log('IndexedDB object store "results" 생성 완료');
      };
      
      request.onsuccess = (event) => {
        const db = event.target.result;
        
        // object store 존재 여부 확인
        if (!db.objectStoreNames.contains('results')) {
          console.error('IndexedDB object store "results"가 생성되지 않았습니다');
          db.close();
          reject(new Error('Object store creation failed'));
          return;
        }
        
        console.log('IndexedDB 초기화 완료');
        db.close();
        resolve();
      };
    });
  };

  const saveToIndexedDB = (key, data) => {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('AgenticAnalysisDB', 2);
      
      request.onerror = () => {
        console.error('IndexedDB 열기 실패:', request.error);
        reject(request.error);
      };
      
      request.onsuccess = (event) => {
        const db = event.target.result;
        
        try {
          // object store 존재 여부 확인
          if (!db.objectStoreNames.contains('results')) {
            console.error('IndexedDB object store "results"를 찾을 수 없습니다');
            db.close();
            reject(new Error('Object store "results" not found'));
            return;
          }
          
          const transaction = db.transaction(['results'], 'readwrite');
          const store = transaction.objectStore('results');
          
          const putRequest = store.put({ 
            id: key, 
            data: data, 
            timestamp: new Date().toISOString() 
          });
          
          putRequest.onsuccess = () => {
            console.log(`IndexedDB에 데이터 저장 성공: ${key}`);
          };
          
          putRequest.onerror = () => {
            console.error('IndexedDB 데이터 저장 실패:', putRequest.error);
          };
          
          transaction.oncomplete = () => {
            db.close();
            resolve();
          };
          
          transaction.onerror = () => {
            console.error('IndexedDB 트랜잭션 오류:', transaction.error);
            db.close();
            reject(transaction.error);
          };
          
        } catch (error) {
          console.error('IndexedDB 트랜잭션 생성 오류:', error);
          db.close();
          reject(error);
        }
      };
    });
  };

  const loadFromIndexedDB = (key) => {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('AgenticAnalysisDB', 2);
      
      request.onerror = () => {
        console.error('IndexedDB 열기 실패:', request.error);
        resolve(null); // 오류 시 null 반환 (reject 대신)
      };
      
      request.onsuccess = (event) => {
        const db = event.target.result;
        
        try {
          // object store 존재 여부 확인
          if (!db.objectStoreNames.contains('results')) {
            console.warn('IndexedDB object store "results"를 찾을 수 없습니다 - 데이터 없음');
            db.close();
            resolve(null);
            return;
          }
          
          const transaction = db.transaction(['results'], 'readonly');
          const store = transaction.objectStore('results');
          
          const getRequest = store.get(key);
          
          getRequest.onsuccess = () => {
            const result = getRequest.result;
            if (result) {
              console.log(`IndexedDB에서 데이터 로드 성공: ${key}`);
              resolve(result.data);
            } else {
              console.log(`IndexedDB에 데이터 없음: ${key}`);
              resolve(null);
            }
          };
          
          getRequest.onerror = () => {
            console.error('IndexedDB 데이터 로드 실패:', getRequest.error);
            resolve(null);
          };
          
          transaction.oncomplete = () => {
            db.close();
          };
          
          transaction.onerror = () => {
            console.error('IndexedDB 트랜잭션 오류:', transaction.error);
            db.close();
            resolve(null);
          };
          
        } catch (error) {
          console.error('IndexedDB 트랜잭션 생성 오류:', error);
          db.close();
          resolve(null);
        }
      };
    });
  };

const App = () => {
  const [user, setUser] = useState(getStoredUser());
  const [selectedKey, setSelectedKey] = useState('home');
  const [loading, setLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState(null);

  // 전역 배치 분석 결과 상태 (페이지 간 공유)
  const [globalBatchResults, setGlobalBatchResults] = useState(null);
  const [lastAnalysisTimestamp, setLastAnalysisTimestamp] = useState(null);
  const [dataLoaded] = useState(true); // 데이터 로딩 상태를 기본적으로 활성화
  const isInitializedRef = useRef(false); // 초기화 중복 방지

  const isAdmin = user?.role === 'admin';

  // 전역 에러 핸들러
  useEffect(() => {
    const handleUnhandledRejection = (event) => {
      // Chrome 확장 프로그램 관련 오류는 완전히 무시
      if (event.reason && event.reason.message && 
          (event.reason.message.includes('Extension context invalidated') ||
           event.reason.message.includes('message channel closed') ||
           event.reason.message.includes('disconnected port object') ||
           event.reason.message.includes('Attempting to use a disconnected port') ||
           event.reason.message.includes('Could not establish connection') ||
           event.reason.message.includes('SecretSessionError'))) {
        event.preventDefault();
        return;
      }
      
      // Chrome extension URL 관련 오류도 무시
      if (event.reason && event.reason.stack && 
          event.reason.stack.includes('chrome-extension://')) {
        event.preventDefault();
        return;
      }
      
      console.error('Unhandled promise rejection:', event.reason);
      notification.error({
        message: '예상치 못한 오류',
        description: '시스템에서 예상치 못한 오류가 발생했습니다.',
        duration: 4.5,
      });
    };

    const handleError = (event) => {
      // Chrome 확장 프로그램 관련 오류는 완전히 무시
      if (event.error && event.error.message && 
          (event.error.message.includes('Extension context invalidated') ||
           event.error.message.includes('message channel closed') ||
           event.error.message.includes('disconnected port object') ||
           event.error.message.includes('Attempting to use a disconnected port') ||
           event.error.message.includes('Could not establish connection') ||
           event.error.message.includes('SecretSessionError'))) {
        event.preventDefault();
        return;
      }
      
      // Chrome extension URL 관련 오류도 무시
      if (event.error && event.error.stack && 
          event.error.stack.includes('chrome-extension://')) {
        event.preventDefault();
        return;
      }
      
      console.error('Global error:', event.error);
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    window.addEventListener('error', handleError);

    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
      window.removeEventListener('error', handleError);
    };
  }, []);

  // 공통 메뉴 (전체 사용자)
  const commonMenuItems = [
    { key: 'home', icon: <HomeOutlined />, label: '인원현황' },
    { key: 'insights', icon: <BulbOutlined />, label: '인사이트' },
    { key: 'risk-factors', icon: <WarningOutlined />, label: '위험 요인' },
    { key: 'intervention', icon: <AimOutlined />, label: '개입 전략' },
    { key: 'report-generation', icon: <FileTextOutlined />, label: '보고서 출력' },
  ];

  // Admin 전용 메뉴
  const adminOnlyItems = [
    { type: 'divider' },
    { key: 'admin-group', icon: <SettingOutlined />, label: 'Admin', type: 'group', children: [
      { key: 'batch', icon: <RobotOutlined />, label: '배치 분석' },
      { key: 'group-statistics', icon: <TeamOutlined />, label: '단체 통계' },
      { key: 'cognita', icon: <ApiOutlined />, label: '개별 관계분석' },
      { key: 'post-analysis', icon: <ExperimentOutlined />, label: '사후 분석' },
      { key: 'admin-settings', icon: <SettingOutlined />, label: '관리자 설정' },
    ]},
  ];

  const menuItems = isAdmin ? [...commonMenuItems, ...adminOnlyItems] : commonMenuItems;

  // 서버 상태 확인 및 IndexedDB/localStorage에서 배치 결과 복원
  useEffect(() => {
    // 이미 초기화되었으면 중복 실행 방지
    if (isInitializedRef.current) {
      return;
    }
    isInitializedRef.current = true;
    
    checkServerStatus();
    
    // IndexedDB 우선, localStorage 백업으로 배치 분석 결과 복원
    const loadInitialData = async () => {
      try {
        // 0. IndexedDB 초기화 먼저 시도
        console.log('🔧 IndexedDB 초기화 중...');
        try {
          await initializeIndexedDB();
          console.log('✅ IndexedDB 초기화 완료');
        } catch (initError) {
          console.warn('IndexedDB 초기화 실패, localStorage만 사용:', initError.message);
        }
        
        // 1. IndexedDB에서 먼저 시도
        console.log('🔍 IndexedDB에서 데이터 로드 시도...');
        const indexedResults = await loadFromIndexedDB('batchAnalysisResults');
        const indexedTimestamp = await loadFromIndexedDB('lastAnalysisTimestamp');
        
        if (indexedResults && Array.isArray(indexedResults) && indexedResults.length > 0) {
          // 배열을 올바른 구조로 감싸서 저장
          const batchResultStructure = {
            success: true,
            results: indexedResults,
            total_employees: indexedResults.length,
            completed_employees: indexedResults.length
          };
          setGlobalBatchResults(batchResultStructure);
          console.log('✅ IndexedDB에서 배치 분석 결과 복원:', indexedResults.length, '개');
          
          if (indexedTimestamp) {
            setLastAnalysisTimestamp(indexedTimestamp);
          }
          return; // IndexedDB에서 성공적으로 로드했으면 localStorage는 시도하지 않음
        }
        
        // 2. IndexedDB에 데이터가 없으면 localStorage에서 시도
        console.log('🔍 IndexedDB에 데이터 없음, localStorage에서 시도...');
        
        // 먼저 일반 저장 방식 확인
        const storedResults = localStorage.getItem('batchAnalysisResults');
        const storedTimestamp = localStorage.getItem('lastAnalysisTimestamp');
        
        if (storedResults && storedTimestamp) {
          let parsedResults = null;
          try {
            parsedResults = JSON.parse(storedResults);
            // 배열인지 확인
            if (Array.isArray(parsedResults)) {
              // 배열을 올바른 구조로 감싸서 저장
              const batchResultStructure = {
                success: true,
                results: parsedResults,
                total_employees: parsedResults.length,
                completed_employees: parsedResults.length
              };
              setGlobalBatchResults(batchResultStructure);
              setLastAnalysisTimestamp(storedTimestamp);
              console.log('✅ localStorage에서 배치 분석 결과 복원 (일반 저장):', parsedResults.length + '명');
            } else if (parsedResults && parsedResults.results) {
              // 이미 올바른 구조인 경우
              setGlobalBatchResults(parsedResults);
              setLastAnalysisTimestamp(storedTimestamp);
              console.log('✅ localStorage에서 배치 분석 결과 복원 (구조 유지)');
            } else {
              console.warn('localStorage의 배치 결과가 올바르지 않습니다:', typeof parsedResults);
              // 객체인 경우 배열로 감싸서 구조 생성
              const batchResultStructure = {
                success: true,
                results: [parsedResults],
                total_employees: 1,
                completed_employees: 1
              };
              setGlobalBatchResults(batchResultStructure);
              setLastAnalysisTimestamp(storedTimestamp);
              console.log('✅ localStorage에서 배치 분석 결과 복원 (객체를 구조로 변환)');
            }
            
            // localStorage 데이터를 IndexedDB로 마이그레이션
            try {
              const dataToMigrate = Array.isArray(parsedResults) ? parsedResults : [parsedResults];
              await saveToIndexedDB('batchAnalysisResults', dataToMigrate);
              await saveToIndexedDB('lastAnalysisTimestamp', storedTimestamp);
              console.log('✅ localStorage 데이터를 IndexedDB로 마이그레이션 완료');
            } catch (migrationError) {
              console.warn('IndexedDB 마이그레이션 실패:', migrationError.message);
            }
          } catch (parseError) {
            console.error('localStorage 데이터 파싱 실패:', parseError);
          }
        } else {
          // 청크 저장 방식 확인
          const metadata = localStorage.getItem('batchAnalysisMetadata');
          if (metadata) {
            const meta = JSON.parse(metadata);
            if (meta.storage_type === 'chunked') {
              console.log(`📦 청크 데이터 복원 시작: ${meta.total_chunks}개 청크`);
              
              const allResults = [];
              for (let i = 0; i < meta.total_chunks; i++) {
                const chunkData = localStorage.getItem(`batchAnalysisResults_chunk_${i}`);
                if (chunkData) {
                  try {
                    const parsedChunk = JSON.parse(chunkData);
                    // 배열인지 확인 후 스프레드 연산자 사용
                    if (Array.isArray(parsedChunk)) {
                      allResults.push(...parsedChunk);
                    } else {
                      console.warn(`청크 ${i}가 배열이 아닙니다:`, typeof parsedChunk);
                      // 객체인 경우 배열로 감싸서 추가
                      allResults.push(parsedChunk);
                    }
                  } catch (parseError) {
                    console.error(`청크 ${i} 파싱 실패:`, parseError);
                  }
                }
              }
              
              if (allResults.length > 0) {
                // 배열을 올바른 구조로 감싸서 저장
                const batchResultStructure = {
                  success: true,
                  results: allResults,
                  total_employees: allResults.length,
                  completed_employees: allResults.length
                };
                setGlobalBatchResults(batchResultStructure);
                setLastAnalysisTimestamp(meta.timestamp);
                console.log(`✅ 청크 데이터 복원 완료: ${allResults.length}명`);
                
                // 청크 데이터도 IndexedDB로 마이그레이션
                try {
                  await saveToIndexedDB('batchAnalysisResults', allResults);
                  await saveToIndexedDB('lastAnalysisTimestamp', meta.timestamp);
                  console.log('✅ 청크 데이터를 IndexedDB로 마이그레이션 완료');
                } catch (migrationError) {
                  console.warn('청크 데이터 마이그레이션 실패:', migrationError.message);
                }
              }
            }
          } else {
            // 요약 데이터만 있는 경우
            const summaryData = localStorage.getItem('batchAnalysisResultsSummary');
            if (summaryData) {
              const summary = JSON.parse(summaryData);
              console.log('요약 데이터만 복원됨:', summary);
              setLastAnalysisTimestamp(summary.timestamp);
            }
          }
        }
        
      } catch (error) {
        console.error('배치 결과 복원 실패:', error);
        
        // 최후의 수단으로 localStorage 직접 시도
        try {
          const savedResults = localStorage.getItem('batchAnalysisResults');
          if (savedResults) {
            const parsedResults = JSON.parse(savedResults);
            if (Array.isArray(parsedResults) && parsedResults.length > 0) {
              // 배열을 올바른 구조로 감싸서 저장
              const batchResultStructure = {
                success: true,
                results: parsedResults,
                total_employees: parsedResults.length,
                completed_employees: parsedResults.length
              };
              setGlobalBatchResults(batchResultStructure);
              console.log('✅ 최후 수단으로 localStorage에서 복원:', parsedResults.length, '개');
            } else if (parsedResults && parsedResults.results) {
              // 이미 올바른 구조인 경우
              setGlobalBatchResults(parsedResults);
              console.log('✅ 최후 수단으로 localStorage에서 복원 (구조 유지)');
            } else if (parsedResults && typeof parsedResults === 'object') {
              // 객체인 경우 배열로 감싸서 구조 생성
              const batchResultStructure = {
                success: true,
                results: [parsedResults],
                total_employees: 1,
                completed_employees: 1
              };
              setGlobalBatchResults(batchResultStructure);
              console.log('✅ 최후 수단으로 localStorage에서 복원 (객체를 구조로 변환)');
            }
          }
        } catch (fallbackError) {
          console.error('모든 데이터 복원 방법 실패:', fallbackError.message);
        }
      }
    };
    
    loadInitialData();
  }, []); // 빈 의존성 배열로 한 번만 실행

  const checkServerStatus = async () => {
    try {
      const status = await apiService.checkHealth();
      setServerStatus(status);
      notification.success({
        message: '서버 연결 성공',
        description: 'Agentic AI System이 정상적으로 작동 중입니다.',
        duration: 3,
      });
    } catch (error) {
      console.error('서버 상태 확인 실패:', error);
      setServerStatus(null);
      notification.error({
        message: '서버 연결 실패',
        description: 'Agentic AI System 백엔드 서버가 실행되지 않았습니다. 서버를 시작한 후 새로고침하세요.',
        duration: 0,
      });
    }
  };


  // 로딩 상태 관리
  const setGlobalLoading = (isLoading) => {
    setLoading(isLoading);
  };

  // 현재 선택된 컴포넌트 렌더링
  // 배치 분석 결과 업데이트 함수
  const updateBatchResults = async (results) => {
    setGlobalBatchResults(results);
    const timestamp = new Date().toISOString();
    setLastAnalysisTimestamp(timestamp);
    
    // IndexedDB 우선 시도, 실패 시 localStorage 사용
    try {
      // IndexedDB 우선 시도
      await saveToIndexedDB('batchAnalysisResults', results);
      await saveToIndexedDB('lastAnalysisTimestamp', timestamp);
      console.log('✅ IndexedDB에 배치 분석 결과 저장 완료');
      return; // 성공하면 LocalStorage 시도하지 않음
    } catch (indexedDBError) {
      console.warn('IndexedDB 저장 실패, LocalStorage로 대체:', indexedDBError.message);
      
      // LocalStorage 백업 시도
      try {
        localStorage.setItem('batchAnalysisResults', JSON.stringify(results));
        localStorage.setItem('lastAnalysisTimestamp', timestamp);
        console.log('배치 분석 결과 전역 업데이트:', results);
      } catch (error) {
        if (error.name === 'QuotaExceededError') {
        console.warn('LocalStorage 용량 초과 - 기존 데이터를 정리하고 청크 단위로 분할 저장합니다.');
        try {
          // 기존 데이터 완전 정리
          const keysToRemove = [];
          for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && (key.startsWith('batchAnalysisResults') || key.startsWith('batch_chunk_'))) {
              keysToRemove.push(key);
            }
          }
          keysToRemove.forEach(key => localStorage.removeItem(key));
          console.log(`🧹 기존 배치 데이터 ${keysToRemove.length}개 항목 정리 완료`);
          
          // 데이터 구조 확인 및 안전한 청크 분할
          const resultArray = results.results || results.data || (Array.isArray(results) ? results : []);
          
          if (!Array.isArray(resultArray) || resultArray.length === 0) {
            console.error('유효하지 않은 결과 데이터 구조:', results);
            throw new Error('Invalid results structure');
          }
          
          // 데이터 압축 및 최소화
          const compressedArray = resultArray.map(item => {
            // 필수 정보만 추출하여 크기 최소화
            return {
              id: item.employee_number || item.employee_id || item.id,
              dept: item.department || 'Unknown',
              risk: item.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 0,
              level: item.risk_level || 'unknown'
            };
          });
          
          // 매우 작은 청크 크기로 설정
          const chunkSize = 20; // 고정된 작은 크기
          const chunks = [];
          
          console.log(`📦 데이터 압축: ${resultArray.length}개 → 압축률 ${Math.round((JSON.stringify(compressedArray).length / JSON.stringify(resultArray).length) * 100)}%`);
          
          for (let i = 0; i < compressedArray.length; i += chunkSize) {
            chunks.push(compressedArray.slice(i, i + chunkSize));
          }
          
          // 각 청크를 개별 키로 저장 (안전한 저장)
          let savedChunks = 0;
          for (let i = 0; i < chunks.length; i++) {
            try {
              const chunkData = {
                chunk_index: i,
                total_chunks: chunks.length,
                data: chunks[i], // 압축된 데이터
                timestamp: timestamp
              };
              localStorage.setItem(`batchAnalysisResults_chunk_${i}`, JSON.stringify(chunkData));
              savedChunks++;
            } catch (chunkError) {
              console.error(`청크 ${i} 저장 실패:`, chunkError);
              break; // 더 이상 저장할 수 없으면 중단
            }
          }
          
          // 메타데이터 저장
          const metadata = {
            total_employees: resultArray.length,
            saved_employees: Math.min(savedChunks * chunkSize, compressedArray.length),
            total_chunks: chunks.length,
            saved_chunks: savedChunks,
            chunk_size: chunkSize,
            timestamp: timestamp,
            storage_type: 'compressed_chunked',
            compression_ratio: Math.round((JSON.stringify(compressedArray).length / JSON.stringify(resultArray).length) * 100),
            original_structure: {
              has_results: !!results.results,
              has_data: !!results.data,
              is_array: Array.isArray(results)
            }
          };
          
          localStorage.setItem('batchAnalysisMetadata', JSON.stringify(metadata));
          localStorage.setItem('lastAnalysisTimestamp', timestamp);
          console.log(`청크 분할 저장 완료: ${savedChunks}/${chunks.length}개 청크, 총 ${Math.min(savedChunks * chunkSize, resultArray.length)}/${resultArray.length}명`);
        } catch (secondError) {
          console.error('청크 저장도 실패:', secondError);
          // 최후의 수단: 요약 데이터만 저장
          try {
            localStorage.clear();
            
            // 안전한 데이터 추출
            const resultArray = results.results || results.data || (Array.isArray(results) ? results : []);
            
            const summaryResults = {
              total_employees: resultArray.length,
              timestamp: timestamp,
              summary: {
                high_risk: resultArray.filter(r => {
                  const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                  return score && score >= 0.7;
                }).length,
                medium_risk: resultArray.filter(r => {
                  const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                  return score && score >= 0.3 && score < 0.7;
                }).length,
                low_risk: resultArray.filter(r => {
                  const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                  return score && score < 0.3;
                }).length
              },
              departments: [...new Set(resultArray.map(r => 
                r.analysis_result?.employee_data?.Department || 
                r.department || 
                'Unknown'
              ))],
              storage_type: 'summary_only',
              error_info: {
                original_error: secondError.message,
                data_structure: {
                  has_results: !!results.results,
                  has_data: !!results.data,
                  is_array: Array.isArray(results),
                  result_count: resultArray.length
                }
              }
            };
            
            localStorage.setItem('batchAnalysisResultsSummary', JSON.stringify(summaryResults));
            localStorage.setItem('lastAnalysisTimestamp', timestamp);
            console.log('⚠️ 요약 데이터만 저장 완료:', summaryResults);
          } catch (finalError) {
            console.error('❌ 모든 저장 방식 실패:', finalError);
            // 최종 실패 시에도 타임스탬프는 저장 시도
            try {
              localStorage.setItem('lastAnalysisTimestamp', timestamp);
              localStorage.setItem('batchAnalysisError', JSON.stringify({
                error: finalError.message,
                timestamp: timestamp,
                attempted_storage: 'all_methods_failed'
              }));
            } catch (timestampError) {
              console.error('타임스탬프 저장도 실패:', timestampError);
            }
          }
        }
        } else {
          console.error('배치 결과 저장 실패:', error);
        }
      }
    }
  };

  const renderContent = () => {
    const commonProps = {
      loading,
      setLoading: setGlobalLoading,
      serverStatus,
      dataLoaded,
      // 전역 배치 결과 전달 
      globalBatchResults,
      lastAnalysisTimestamp,
      updateBatchResults, // 배치 결과 업데이트 함수
    };

    switch (selectedKey) {
      case 'home':
        return <Home {...commonProps} onNavigate={setSelectedKey} />;
      case 'insights':
        return <InsightsPlaceholder />;
      case 'risk-factors':
        return <RiskFactorsPlaceholder />;
      case 'intervention':
        return <InterventionPlaceholder />;
      case 'report-generation':
        return <ReportGeneration {...commonProps} />;
      case 'batch':
        return <BatchAnalysis {...commonProps} onNavigate={setSelectedKey} />;
      case 'group-statistics':
        return <GroupStatistics {...commonProps} />;
      case 'cognita':
        return <RelationshipAnalysis {...commonProps} batchResults={globalBatchResults} />;
      case 'post-analysis':
        return <PostAnalysis {...commonProps} />;
      case 'admin-settings':
        return <AdminSettingsPlaceholder />;
      default:
        return <Home {...commonProps} onNavigate={setSelectedKey} />;
    }
  };

  // 다크모드 상태
  const [themeMode, setThemeMode] = useState(() => localStorage.getItem('pwc_theme') || 'system');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const isDark = themeMode === 'dark' || (themeMode === 'system' && prefersDark);

  // 조직 View 상태
  const [viewMode, setViewMode] = useState('all'); // 'all' | department name
  const departments = ['Research & Development', 'Sales', 'Human Resources'];

  // 테마 변경 시 localStorage 저장
  const cycleTheme = () => {
    const next = themeMode === 'light' ? 'dark' : themeMode === 'dark' ? 'system' : 'light';
    setThemeMode(next);
    localStorage.setItem('pwc_theme', next);
  };

  // 로그인 전이면 로그인 화면 표시
  if (!user) {
    return <Login onLogin={setUser} />;
  }

  // 다크모드 CSS 변수
  const themeVars = isDark ? {
    '--bg': '#1a1a2e', '--card': '#16213e', '--border': '#333',
    '--text': '#e0e0e0', '--sub': '#888', '--header-bg': '#0f3460',
  } : {
    '--bg': '#f3f4f6', '--card': '#fff', '--border': '#eee',
    '--text': '#2d2d2d', '--sub': '#888', '--header-bg': '#fff',
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: themeVars['--bg'],
      color: themeVars['--text'],
      fontFamily: "'Noto Sans KR', system-ui, -apple-system, sans-serif",
      transition: 'background 0.3s, color 0.3s',
    }}>
      {/* ── HEADER ── */}
      <header style={{
        position: 'sticky', top: 0, zIndex: 200,
        background: themeVars['--header-bg'],
        borderBottom: '3px solid #d93954',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 24px', height: 56,
        boxShadow: '0 1px 4px rgba(0,0,0,.08)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <img src={PWC_LOGO} alt="PwC" style={{ height: 28 }} />
          <span style={{ fontSize: 17, fontWeight: 700, color: themeVars['--text'] }}>
            조직 퇴사위험 대시보드
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 13 }}>
          {/* 전사/조직 View 토글 */}
          <div style={{ display: 'flex', border: `1px solid ${isDark ? '#444' : '#ddd'}`, borderRadius: 8, overflow: 'hidden', fontSize: 12 }}>
            <button
              onClick={() => setViewMode('all')}
              style={{
                padding: '5px 14px', border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontWeight: 500,
                background: viewMode === 'all' ? '#d93954' : themeVars['--card'],
                color: viewMode === 'all' ? '#fff' : themeVars['--text'],
              }}
            >전사 View</button>
            {departments.map(d => (
              <button
                key={d}
                onClick={() => setViewMode(d)}
                style={{
                  padding: '5px 10px', border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontWeight: 500,
                  borderLeft: `1px solid ${isDark ? '#444' : '#ddd'}`,
                  background: viewMode === d ? '#d93954' : themeVars['--card'],
                  color: viewMode === d ? '#fff' : themeVars['--text'],
                  fontSize: 11,
                }}
              >{d.replace('Research & Development', 'R&D').replace('Human Resources', 'HR')}</button>
            ))}
          </div>

          {/* 다크모드 토글 */}
          <button onClick={cycleTheme} style={{
            border: `1px solid ${isDark ? '#444' : '#ddd'}`, borderRadius: 6,
            padding: '4px 10px', cursor: 'pointer', fontSize: 14,
            background: themeVars['--card'], color: themeVars['--text'],
          }}>
            {themeMode === 'light' ? '☀️' : themeMode === 'dark' ? '🌙' : '💻'}
          </button>

          {/* 유저 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{
              width: 32, height: 32, borderRadius: '50%',
              background: '#d93954', color: '#fff',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 12, fontWeight: 600,
            }}>{user.initials}</div>
            <span style={{ fontWeight: 500, color: themeVars['--sub'] }}>{user.name}</span>
            <span style={{
              fontSize: 11, padding: '2px 8px', borderRadius: 4,
              background: isAdmin ? '#fde8ec' : '#e8f0fe',
              color: isAdmin ? '#d93954' : '#2563eb', fontWeight: 600,
            }}>{isAdmin ? 'ADMIN' : 'HR'}</span>
          </div>

          <button onClick={() => { logout(); setUser(null); }} style={{
            border: `1px solid ${isDark ? '#444' : '#ddd'}`, background: themeVars['--card'],
            borderRadius: 6, padding: '5px 12px', cursor: 'pointer',
            fontSize: 12, fontFamily: 'inherit', color: themeVars['--sub'], fontWeight: 500,
          }}>로그아웃</button>
        </div>
      </header>

      {/* ── TAB NAV ── */}
      <nav style={{
        background: themeVars['--card'],
        display: 'flex', padding: '0 24px',
        borderBottom: `1px solid ${themeVars['--border']}`,
        boxShadow: '0 1px 3px rgba(0,0,0,.04)',
        overflowX: 'auto', position: 'sticky', top: 56, zIndex: 190,
      }}>
        {menuItems.filter(i => i.key).map(item => (
          <button
            key={item.key}
            onClick={() => setSelectedKey(item.key)}
            style={{
              padding: '12px 22px', fontSize: 14, fontWeight: selectedKey === item.key ? 700 : 500,
              color: selectedKey === item.key ? '#d93954' : themeVars['--sub'],
              cursor: 'pointer', border: 'none', background: 'none',
              borderBottom: selectedKey === item.key ? '3px solid #d93954' : '3px solid transparent',
              whiteSpace: 'nowrap', fontFamily: 'inherit',
              transition: 'all .2s',
            }}
          >{item.label}</button>
        ))}
      </nav>

      {/* 조직 View 표시 */}
      {viewMode !== 'all' && (
        <div style={{
          background: isDark ? '#1e3a5f' : '#fef7f8',
          padding: '8px 24px', fontSize: 13, fontWeight: 600,
          color: '#d93954', borderBottom: `1px solid ${themeVars['--border']}`,
        }}>
          📂 조직 View: {viewMode} | 해당 조직의 데이터만 필터링됩니다
        </div>
      )}

      {/* ── MAIN CONTENT ── */}
      <main style={{
        padding: '20px 24px 32px',
        maxWidth: 1600, margin: '0 auto',
      }}>
        {renderContent()}
      </main>

      <footer style={{ textAlign: 'center', padding: 20, fontSize: 11, color: themeVars['--sub'] }}>
        &copy; 2025 PwC Consulting. Agentic AI 기반 선제적 퇴사위험 예측 및 관리시스템.
      </footer>
    </div>
  );
};

export default App;
