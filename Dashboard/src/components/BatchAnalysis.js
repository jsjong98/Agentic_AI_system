import React, { useState } from 'react';
import {
  Card,
  Button,
  Upload,
  message,
  Progress,
  Typography,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Alert,
  Space,
  Input,
  Slider
} from 'antd';
import {
  UploadOutlined,
  FileTextOutlined,
  ApiOutlined,
  BarChartOutlined,
  RocketOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  DownloadOutlined,
  DashboardOutlined,
  TeamOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { Dragger } = Upload;

const BatchAnalysis = ({
  loading,
  setLoading,
  serverStatus
}) => {
  const [analysisResults, setAnalysisResults] = useState(null);
  const [analysisProgress, setAnalysisProgress] = useState({
    structura: 0,
    cognita: 0,
    chronos: 0,
    sentio: 0,
    agora: 0,
    overall: 0
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [employeeData, setEmployeeData] = useState([]);
  
  // 각 에이전트별 데이터 상태
  const [agentFiles, setAgentFiles] = useState({
    structura: null,
    chronos: null,
    sentio: null,
    agora: null
  });
  
  // Neo4j 연결 설정
  const [neo4jConfig, setNeo4jConfig] = useState({
    uri: 'bolt://44.212.67.74:7687',
    username: 'neo4j',
    password: 'legs-augmentations-cradle'
  });
  const [neo4jConnected, setNeo4jConnected] = useState(false);
  const [neo4jTesting, setNeo4jTesting] = useState(false);
  
  
  // Integration 설정 (노트북 분석 결과 기반)
  const [integrationConfig, setIntegrationConfig] = useState({
    structura_weight: 0.3216,  // Bayesian Optimization 최적값
    cognita_weight: 0.1000,
    chronos_weight: 0.3690,
    sentio_weight: 0.1000,
    agora_weight: 0.1094,
    high_risk_threshold: 0.7,
    medium_risk_threshold: 0.4,
    // 개별 에이전트 임계값 (Threshold_setting.ipynb 결과)
    structura_threshold: 0.899000,
    cognita_threshold: 0.475200,
    chronos_threshold: 0.010100,
    sentio_threshold: 0.465800,
    agora_threshold: 0.245800
  });


  // 에이전트별 파일 업로드 처리
  const handleAgentFileUpload = async (file, agentType) => {
    const isCSV = file.type === 'text/csv' || file.name.endsWith('.csv');
    if (!isCSV) {
      message.error('CSV 파일만 업로드 가능합니다.');
      return false;
    }

    // 파일 크기 제한 (Chronos는 500MB, 나머지는 10MB)
    const maxSize = agentType === 'chronos' ? 500 : 10;
    const isLtMaxSize = file.size / 1024 / 1024 < maxSize;
    if (!isLtMaxSize) {
      message.error(`파일 크기는 ${maxSize}MB 이하여야 합니다.`);
      return false;
    }

    try {
      setLoading(true);
      
      // CSV 파일 읽기 및 검증
      const text = await file.text();
      const lines = text.split('\n').filter(line => line.trim());
      
      if (lines.length < 2) {
        message.error('유효한 CSV 데이터가 없습니다.');
        return false;
      }

      // 에이전트별 필수 컬럼 검증
      const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
      const requiredColumns = getRequiredColumns(agentType);
      const missingColumns = requiredColumns.filter(col => !headers.includes(col));
      
      if (missingColumns.length > 0) {
        message.error(`필수 컬럼이 누락되었습니다: ${missingColumns.join(', ')}`);
        return false;
      }

      // 파일 저장
      setAgentFiles(prev => ({
        ...prev,
        [agentType]: file
      }));

      // Structura 파일인 경우 직원 데이터도 파싱
      if (agentType === 'structura') {
        const employees = parseEmployeeData(lines, headers);
        setEmployeeData(employees);
      }

      message.success(`${agentType} 데이터를 로드했습니다.`);
      return false; // Ant Design Upload 컴포넌트의 자동 업로드 방지
      
    } catch (error) {
      console.error(`${agentType} 파일 업로드 실패:`, error);
      message.error(`${agentType} 파일 업로드 실패: ${error.message}`);
      return false;
    } finally {
      setLoading(false);
    }
  };

  // 에이전트별 필수 컬럼 정의
  const getRequiredColumns = (agentType) => {
    const columnMap = {
      structura: ['EmployeeNumber', 'Age', 'JobRole', 'Department'],
      chronos: ['employee_id', 'date', 'work_focused_ratio', 'meeting_collaboration_ratio'],
      sentio: ['EmployeeNumber', 'SELF_REVIEW_text', 'PEER_FEEDBACK_text', 'WEEKLY_SURVEY_text'],
      agora: ['EmployeeNumber', 'Age', 'JobRole', 'Department'] // Structura와 동일한 데이터 사용
    };
    return columnMap[agentType] || [];
  };

  // 직원 데이터 파싱
  const parseEmployeeData = (lines, headers) => {
    const employees = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
      const employee = {};
      
      headers.forEach((header, index) => {
        let value = values[index] || '';
        
        // 숫자 컬럼 처리
        if (['Age', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance'].includes(header)) {
          value = parseFloat(value) || 0;
        }
        
        employee[header] = value;
      });
      
      if (employee.EmployeeNumber) {
        employees.push(employee);
      }
    }
    return employees;
  };

  // Neo4j 연결 테스트
  const testNeo4jConnection = async () => {
    if (!neo4jConfig.uri || !neo4jConfig.username || !neo4jConfig.password) {
      message.error('Neo4j 연결 정보를 모두 입력해주세요.');
      return;
    }

    setNeo4jTesting(true);
    try {
      const response = await fetch('/api/cognita/setup/neo4j', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(neo4jConfig)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('Neo4j 연결 테스트 응답:', result);
      
      // 응답 형식 통일 (success 필드 확인)
      if (result.success) {
        setNeo4jConnected(true);
        message.success('Neo4j 연결 성공!');
      } else {
        setNeo4jConnected(false);
        const errorMsg = result.error || result.message || '알 수 없는 오류';
        message.error(`Neo4j 연결 실패: ${errorMsg}`);
        console.error('Neo4j 연결 실패 상세:', result);
      }
    } catch (error) {
      console.error('Neo4j 연결 테스트 실패:', error);
      setNeo4jConnected(false);
      message.error(`Neo4j 연결 테스트 실패: ${error.message}`);
    } finally {
      setNeo4jTesting(false);
    }
  };

  // 에이전트 디버깅 정보 조회
  const debugAgents = async () => {
    try {
      console.log('🔍 에이전트 디버깅 정보 조회 시작');
      
      const response = await fetch('/api/agents/debug');
      const debugInfo = await response.json();
      
      console.log('🔍 에이전트 디버깅 정보:', debugInfo);
      
      // Console에 상세 정보 출력
      console.log('📊 에이전트별 상태:');
      Object.entries(debugInfo.agents).forEach(([agentName, info]) => {
        console.log(`  ${agentName}:`);
        console.log(`    - Import 가능: ${info.import_available}`);
        console.log(`    - 초기화됨: ${info.initialized}`);
        console.log(`    - 객체 존재: ${info.agent_object}`);
        if (info.error_message) {
          console.log(`    - 오류: ${info.error_message}`);
        }
      });
      
      message.success('에이전트 디버깅 정보가 Console에 출력되었습니다. F12를 눌러 확인하세요.');
      
    } catch (error) {
      console.error('❌ 에이전트 디버깅 정보 조회 실패:', error);
      message.error(`디버깅 정보 조회 실패: ${error.message}`);
    }
  };

  // 통합 배치 분석 실행 - 데이터 검증 + 순차적 워크플로우 + Integration
  const runBatchAnalysis = async () => {
    // 1. 필수 데이터 검증
    if (!agentFiles.structura) {
      message.error('Structura 데이터(HR 기본 데이터)를 업로드해주세요.');
      return;
    }

    // 2. 가중치 합계 검증
    const weightSum = integrationConfig.structura_weight + 
                     integrationConfig.cognita_weight + 
                     integrationConfig.chronos_weight + 
                     integrationConfig.sentio_weight + 
                     integrationConfig.agora_weight;
    
    if (Math.abs(weightSum - 1.0) > 0.01) {
      message.error(`가중치 합계가 1.0이 되어야 합니다. 현재: ${weightSum.toFixed(2)}`);
      return;
    }

    // 진행률 폴링을 위한 변수들을 함수 스코프에서 선언
    let progressInterval = null;
    let cleanup = null;

    try {
      setIsAnalyzing(true);
      setAnalysisProgress({
        structura: 0,
        cognita: 0,
        chronos: 0,
        sentio: 0,
        agora: 0,
        overall: 0
      });
      
      // 3. 에이전트별 활성화 여부 결정
      const agentConfig = {
        use_structura: !!agentFiles.structura,
        use_cognita: neo4jConnected,
        use_chronos: !!agentFiles.chronos,
        use_sentio: !!agentFiles.sentio,
        use_agora: !!agentFiles.agora || !!agentFiles.structura // Agora는 별도 파일 또는 Structura 데이터 사용
      };

      // Console 로그 추가
      console.log('🚀 배치 분석 시작');
      console.log('📊 직원 데이터:', employeeData.length, '명');
      console.log('🔧 에이전트 설정:', agentConfig);
      console.log('⚙️ 통합 설정:', integrationConfig);
      console.log('🔗 Neo4j 연결 상태:', neo4jConnected);

      // 4. 실제 진행률 폴링 시작
      progressInterval = setInterval(async () => {
        try {
          const progressResponse = await fetch('/api/analyze/batch/progress');
          if (progressResponse.ok) {
            const progressData = await progressResponse.json();
            
            if (progressData.success) {
              setAnalysisProgress({
                structura: progressData.agent_progress?.structura || 0,
                cognita: progressData.agent_progress?.cognita || 0,
                chronos: progressData.agent_progress?.chronos || 0,
                sentio: progressData.agent_progress?.sentio || 0,
                agora: progressData.agent_progress?.agora || 0,
                overall: progressData.overall_progress || 0
              });
              
              // 분석 완료 시 폴링 중단
              if (progressData.status === 'completed') {
                clearInterval(progressInterval);
              }
            }
          }
        } catch (error) {
          console.error('진행률 조회 실패:', error);
        }
      }, 1000); // 1초마다 폴링
      
      // 컴포넌트 언마운트 시 폴링 정리
      cleanup = () => {
        if (progressInterval) {
          clearInterval(progressInterval);
        }
      };
      
      // 분석 완료 후 정리
      window.addEventListener('beforeunload', cleanup);

      // 5. Supervisor를 통한 배치 분석 API 호출
      const response = await fetch('/api/analyze/batch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          employees: employeeData,
          ...agentConfig,
          integration_config: integrationConfig,
          neo4j_config: neo4jConnected ? neo4jConfig : null,
          agent_files: {
            structura: agentFiles.structura?.name,
            chronos: agentFiles.chronos?.name,
            sentio: agentFiles.sentio?.name,
            agora: agentFiles.agora?.name
          }
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const batchResult = await response.json();
      
      console.log('📥 배치 분석 응답 받음:', batchResult);
      
      if (batchResult.error) {
        console.error('❌ 배치 분석 오류:', batchResult.error);
        throw new Error(batchResult.error);
      }

      // 결과 분석 로깅
      if (batchResult.results) {
        const successCount = batchResult.results.filter(r => r.analysis_result && r.analysis_result.status === 'success').length;
        const failureCount = batchResult.results.filter(r => r.analysis_result && r.analysis_result.status === 'failed').length;
        console.log(`📈 분석 결과: 성공 ${successCount}명, 실패 ${failureCount}명`);
        
        // 실패한 케이스들 상세 로깅
        const failures = batchResult.results.filter(r => r.analysis_result && r.analysis_result.status === 'failed');
        if (failures.length > 0) {
          console.log('❌ 실패한 분석들:');
          failures.forEach((failure, index) => {
            console.log(`  ${index + 1}. 직원 ${failure.employee_number}:`, failure.analysis_result);
          });
        }
      }

      // 폴링 정리
      if (progressInterval) {
        clearInterval(progressInterval);
      }
      window.removeEventListener('beforeunload', cleanup);

      // 5. Integration 보고서 생성
      const reportResponse = await fetch('/api/integration/report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysis_results: batchResult.results,
          report_options: {
            include_recommendations: true,
            include_risk_analysis: true,
            integration_config: integrationConfig
          }
        })
      });

      if (!reportResponse.ok) {
        throw new Error(`Integration 보고서 생성 실패: ${reportResponse.status}`);
      }

      const reportResult = await reportResponse.json();
      
      if (reportResult.error) {
        throw new Error(reportResult.error);
      }

      // 6. 최종 결과 설정
      setAnalysisResults({
        ...batchResult,
        integration_report: reportResult.report,
        report_metadata: reportResult.metadata
      });
      
      setAnalysisProgress(prev => ({ ...prev, overall: 100 }));

      message.success(`통합 배치 분석 완료! ${batchResult.completed_employees}명의 직원 분석 및 Integration 보고서 생성이 완료되었습니다.`);

    } catch (error) {
      console.error('❌ 통합 배치 분석 실패:', error);
      console.error('❌ 오류 스택:', error.stack);
      
      // 네트워크 오류인지 확인
      if (error.message.includes('fetch')) {
        console.error('🌐 네트워크 연결 문제가 발생했습니다. 백엔드 서버가 실행 중인지 확인하세요.');
        message.error('네트워크 연결 문제가 발생했습니다. 백엔드 서버 상태를 확인하세요.');
      } else {
        message.error(`통합 배치 분석 실패: ${error.message}`);
      }
    } finally {
      setIsAnalyzing(false);
      
      // 폴링 정리 (에러 발생 시에도)
      try {
        if (progressInterval) {
          clearInterval(progressInterval);
        }
        if (cleanup) {
          window.removeEventListener('beforeunload', cleanup);
        }
      } catch (cleanupError) {
        console.log('폴링 정리 중 오류:', cleanupError);
      }
    }
  };

  // 위험도 레벨 계산 함수
  const calculateRiskLevel = (score) => {
    if (score >= 0.7) return 'HIGH';
    if (score >= 0.4) return 'MEDIUM';
    return 'LOW';
  };

  // 결과 내보내기 함수
  const exportResults = (format) => {
    if (!analysisResults) {
      message.error('내보낼 분석 결과가 없습니다.');
      return;
    }

    try {
      if (format === 'excel') {
        // Excel 형태로 데이터 준비
        const exportData = analysisResults.results.map(result => ({
          '직원번호': result.employee_number,
          '통합점수': result.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score 
            ? (result.analysis_result.combined_analysis.integrated_assessment.overall_risk_score * 100).toFixed(1) + '%' 
            : 'N/A',
          '위험도': result.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score 
            ? calculateRiskLevel(result.analysis_result.combined_analysis.integrated_assessment.overall_risk_score)
            : 'N/A',
          'Structura점수': result.analysis_result?.structura_result?.prediction?.attrition_probability 
            ? (result.analysis_result.structura_result.prediction.attrition_probability * 100).toFixed(1) + '%' 
            : 'N/A',
          'Cognita점수': result.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score 
            ? (result.analysis_result.cognita_result.risk_analysis.overall_risk_score * 100).toFixed(1) + '%' 
            : 'N/A',
          'Chronos점수': result.analysis_result?.chronos_result?.trend_score 
            ? (result.analysis_result.chronos_result.trend_score * 100).toFixed(1) + '%' 
            : 'N/A',
          'Sentio점수': result.analysis_result?.sentio_result?.sentiment_score !== undefined
            ? (Math.abs(result.analysis_result.sentio_result.sentiment_score) * 100).toFixed(1) + '%' 
            : 'N/A',
          'Agora점수': result.analysis_result?.agora_result?.market_analysis?.market_pressure_index 
            ? (result.analysis_result.agora_result.market_analysis.market_pressure_index * 100).toFixed(1) + '%' 
            : 'N/A',
          '상태': result.error ? '오류' : '완료'
        }));

        // CSV 형태로 변환
        const headers = Object.keys(exportData[0]);
        const csvContent = [
          headers.join(','),
          ...exportData.map(row => headers.map(header => row[header]).join(','))
        ].join('\n');

        // 파일 다운로드
        const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `batch_analysis_results_${new Date().toISOString().split('T')[0]}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        message.success('Excel 파일이 다운로드되었습니다.');
      } else if (format === 'pdf') {
        // PDF 보고서 생성 요청
        message.info('PDF 보고서 생성 기능은 개발 중입니다.');
      }
    } catch (error) {
      console.error('결과 내보내기 실패:', error);
      message.error('결과 내보내기에 실패했습니다.');
    }
  };

  // 시스템 현황으로 이동
  const navigateToSystemStatus = () => {
    // 분석 결과를 localStorage에 저장
    if (analysisResults) {
      localStorage.setItem('batchAnalysisResults', JSON.stringify(analysisResults));
      message.success('분석 결과가 시스템 현황에 연동되었습니다.');
    }
    // 실제 라우팅은 상위 컴포넌트에서 처리
    message.info('시스템 현황 메뉴로 이동하세요.');
  };

  // 시각화 대시보드로 이동
  const navigateToVisualization = () => {
    // 분석 결과를 localStorage에 저장
    if (analysisResults) {
      localStorage.setItem('batchAnalysisResults', JSON.stringify(analysisResults));
      message.success('분석 결과가 시각화 대시보드에 연동되었습니다.');
    }
    // 실제 라우팅은 상위 컴포넌트에서 처리
    message.info('시각화 대시보드 메뉴로 이동하세요.');
  };

  // 관계 분석으로 이동
  const navigateToRelationshipAnalysis = () => {
    // 분석 결과를 localStorage에 저장
    if (analysisResults) {
      localStorage.setItem('batchAnalysisResults', JSON.stringify(analysisResults));
      message.success('분석 결과가 관계 분석에 연동되었습니다.');
    }
    // 실제 라우팅은 상위 컴포넌트에서 처리
    message.info('🕸️ 개별 관계분석 메뉴로 이동하세요.');
  };

  // 결과 테이블 컬럼 정의
  const resultColumns = [
    {
      title: '직원번호',
      dataIndex: 'employee_number',
      key: 'employee_number',
      width: 100,
      sorter: (a, b) => parseInt(a.employee_number) - parseInt(b.employee_number),
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: '통합 점수',
      key: 'overall_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 0;
        const scoreB = b.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: '위험도',
      key: 'risk_level',
      width: 100,
      render: (_, record) => {
        const score = record.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
        const riskLevel = score ? calculateRiskLevel(score) : 'N/A';
        const color = riskLevel === 'HIGH' ? 'red' : riskLevel === 'MEDIUM' ? 'orange' : riskLevel === 'LOW' ? 'green' : 'default';
        return <Tag color={color}>{riskLevel}</Tag>;
      },
      filters: [
        { text: '고위험군', value: 'HIGH' },
        { text: '중위험군', value: 'MEDIUM' },
        { text: '저위험군', value: 'LOW' },
      ],
      onFilter: (value, record) => {
        const score = record.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
        const riskLevel = score ? calculateRiskLevel(score) : 'N/A';
        return riskLevel === value;
      },
    },
    {
      title: 'Structura 점수',
      key: 'structura_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.structura_result?.prediction?.attrition_probability;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.structura_result?.prediction?.attrition_probability || 0;
        const scoreB = b.analysis_result?.structura_result?.prediction?.attrition_probability || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Cognita 점수',
      key: 'cognita_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score || 0;
        const scoreB = b.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Chronos 점수',
      key: 'chronos_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.chronos_result?.trend_score;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.chronos_result?.trend_score || 0;
        const scoreB = b.analysis_result?.chronos_result?.trend_score || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Sentio 점수',
      key: 'sentio_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.sentio_result?.sentiment_score;
        return score !== undefined ? (Math.abs(score) * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = Math.abs(a.analysis_result?.sentio_result?.sentiment_score || 0);
        const scoreB = Math.abs(b.analysis_result?.sentio_result?.sentiment_score || 0);
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Agora 점수',
      key: 'agora_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.agora_result?.market_analysis?.market_pressure_index;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.agora_result?.market_analysis?.market_pressure_index || 0;
        const scoreB = b.analysis_result?.agora_result?.market_analysis?.market_pressure_index || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: '상태',
      key: 'status',
      width: 80,
      render: (_, record) => {
        if (record.error) {
          return <Tag color="red">오류</Tag>;
        }
        return <Tag color="green">완료</Tag>;
      }
    }
  ];


  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <RocketOutlined /> 배치 분석
      </Title>
      
      <Paragraph>
        HR 데이터 파일을 업로드하여 전체 직원에 대한 종합적인 이직 위험도 분석을 수행합니다.
        Supervisor가 각 에이전트를 순차적으로 실행하여 정확한 분석 결과를 제공합니다.
      </Paragraph>

      <Alert
        message="순차적 에이전트 실행 순서"
        description="Structura (HR 분석) → Cognita (관계 분석) → Chronos (시계열 분석) → Sentio (감정 분석) → Agora (시장 분석)"
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
      />

      <Alert
        message="🔍 디버깅 정보"
        description={
          <div>
            분석 과정에서 문제가 발생하면 브라우저의 개발자 도구(F12) → Console 탭에서 상세한 로그를 확인할 수 있습니다. 에이전트별 실행 상태와 오류 정보가 실시간으로 표시됩니다.
            <div style={{ marginTop: '8px' }}>
              <Button 
                size="small" 
                onClick={debugAgents}
                disabled={loading || isAnalyzing}
              >
                🔍 에이전트 상태 확인
              </Button>
            </div>
          </div>
        }
        type="warning"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Alert
        message="📊 최적화된 임계치 및 가중치 적용"
        description="Threshold_setting.ipynb와 Weight_setting.ipynb 분석 결과를 기반으로 한 최적 설정값이 자동 적용됩니다."
        type="success"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[24, 24]}>
        {/* 파일 업로드 섹션 */}
        {/* 1단계: Structura 데이터 (정형 데이터) */}
        <Col span={12}>
          <Card title="1단계: Structura - 정형 데이터" extra={<FileTextOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="structura_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'structura')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  HR 기본 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: EmployeeNumber, Age, JobRole, Department
                  <br />
                  선택: MonthlyIncome, JobSatisfaction, WorkLifeBalance
                </p>
              </Dragger>
              {agentFiles.structura && (
                <Alert
                  message={`✅ ${agentFiles.structura.name}`}
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 2단계: Cognita 데이터 (Neo4j 연결) */}
        <Col span={12}>
          <Card title="2단계: Cognita - Neo4j 연결" extra={<ApiOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Input
                placeholder="Neo4j URI (예: bolt://localhost:7687)"
                value={neo4jConfig.uri}
                onChange={(e) => setNeo4jConfig({...neo4jConfig, uri: e.target.value})}
                disabled={loading || isAnalyzing}
              />
              <Input
                placeholder="Username (예: neo4j)"
                value={neo4jConfig.username}
                onChange={(e) => setNeo4jConfig({...neo4jConfig, username: e.target.value})}
                disabled={loading || isAnalyzing}
              />
              <Input.Password
                placeholder="Password"
                value={neo4jConfig.password}
                onChange={(e) => setNeo4jConfig({...neo4jConfig, password: e.target.value})}
                disabled={loading || isAnalyzing}
              />
              <Button
                type="dashed"
                onClick={testNeo4jConnection}
                disabled={loading || isAnalyzing}
                loading={neo4jTesting}
              >
                연결 테스트
              </Button>
              {neo4jConnected && (
                <Alert
                  message="✅ Neo4j 연결 성공"
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 3단계: Chronos 데이터 (시계열 데이터) */}
        <Col span={12}>
          <Card title="3단계: Chronos - 시계열 데이터" extra={<BarChartOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="chronos_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'chronos')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  시계열 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: employee_id, date, work_focused_ratio, meeting_collaboration_ratio
                  <br />
                  선택: social_dining_ratio, break_relaxation_ratio, system_login_hours
                  <br />
                  ⚠️ 대용량 파일 지원 (최대 500MB)
                </p>
              </Dragger>
              {agentFiles.chronos && (
                <Alert
                  message={`✅ ${agentFiles.chronos.name} (${(agentFiles.chronos.size/1024/1024).toFixed(1)}MB)`}
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 4단계: Sentio 데이터 (텍스트 데이터) */}
        <Col span={12}>
          <Card title="4단계: Sentio - 텍스트 데이터" extra={<FileTextOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="sentio_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'sentio')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  텍스트/피드백 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: EmployeeNumber, SELF_REVIEW_text, PEER_FEEDBACK_text, WEEKLY_SURVEY_text
                  <br />
                  선택: JobRole, Persona_Code, Persona_Name
                </p>
              </Dragger>
              {agentFiles.sentio && (
                <Alert
                  message={`✅ ${agentFiles.sentio.name}`}
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 5단계: Agora 데이터 (시장 데이터) - 선택사항 */}
        <Col span={12}>
          <Card title="5단계: Agora - 시장 데이터 (선택사항)" extra={<BarChartOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="agora_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'agora')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  시장 분석용 HR 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: EmployeeNumber, Age, JobRole, Department (Structura와 동일)
                  <br />
                  💡 API를 통해 직업별 연봉, 시장 동향 데이터 자동 수집
                </p>
              </Dragger>
              {agentFiles.agora && (
                <Alert
                  message={`✅ ${agentFiles.agora.name}`}
                  type="success"
                  showIcon
                />
              )}
              
              {!agentFiles.agora && agentFiles.structura && (
                <Alert
                  message="Structura 데이터로 Agora 분석 가능"
                  description="별도 파일을 업로드하지 않아도 Structura 데이터를 기반으로 시장 분석을 수행할 수 있습니다."
                  type="info"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 6단계: Integration 설정 (임계치 및 가중치) */}
        <Col span={24}>
          <Card title="6단계: Integration 설정 - 임계치 및 가중치" extra={<SettingOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message="에이전트별 가중치 및 위험도 임계치 설정 (최적화 완료)"
                description="Bayesian Optimization을 통해 도출된 최적 가중치와 F1-Score 기반 최적 임계치가 적용되었습니다. 필요시 수동 조정 가능합니다."
                type="success"
                showIcon
              />
              
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Title level={5}>개별 에이전트 임계치 (F1-Score 최적화)</Title>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Alert
                      message="개별 에이전트 최적 임계치"
                      description="각 에이전트별로 F1-Score를 최대화하는 임계치가 자동 적용됩니다. 사후 분석에서 재계산 가능합니다."
                      type="info"
                      showIcon
                      style={{ marginBottom: 16 }}
                    />
                    
                    <div style={{ background: '#f5f5f5', padding: '12px', borderRadius: '6px' }}>
                      <Text strong>Structura 임계치: </Text>
                      <Text code>{integrationConfig.structura_threshold.toFixed(6)}</Text>
                      <Text type="secondary"> (F1: 0.8306)</Text>
                    </div>
                    
                    <div style={{ background: '#f5f5f5', padding: '12px', borderRadius: '6px' }}>
                      <Text strong>Chronos 임계치: </Text>
                      <Text code>{integrationConfig.chronos_threshold.toFixed(6)}</Text>
                      <Text type="secondary"> (F1: 0.7846)</Text>
                    </div>
                    
                    <div style={{ background: '#f5f5f5', padding: '12px', borderRadius: '6px' }}>
                      <Text strong>Cognita 임계치: </Text>
                      <Text code>{integrationConfig.cognita_threshold.toFixed(6)}</Text>
                      <Text type="secondary"> (F1: 0.2947)</Text>
                    </div>
                    
                    <div style={{ background: '#f5f5f5', padding: '12px', borderRadius: '6px' }}>
                      <Text strong>Agora 임계치: </Text>
                      <Text code>{integrationConfig.agora_threshold.toFixed(6)}</Text>
                      <Text type="secondary"> (F1: 0.2907)</Text>
                    </div>
                    
                    <div style={{ background: '#f5f5f5', padding: '12px', borderRadius: '6px' }}>
                      <Text strong>Sentio 임계치: </Text>
                      <Text code>{integrationConfig.sentio_threshold.toFixed(6)}</Text>
                      <Text type="secondary"> (F1: 0.2892)</Text>
                    </div>
                  </Space>
                </Col>
                
                <Col span={12}>
                  <Title level={5}>에이전트별 가중치 (Bayesian Optimization 최적화)</Title>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text>Structura (정형 데이터): {integrationConfig.structura_weight.toFixed(4)} 
                        <Text type="secondary"> (F1: 0.8306)</Text>
                      </Text>
                      <Slider
                        min={0}
                        max={1}
                        step={0.0001}
                        value={integrationConfig.structura_weight}
                        onChange={(value) => setIntegrationConfig(prev => ({...prev, structura_weight: value}))}
                        disabled={loading || isAnalyzing}
                      />
                    </div>
                    <div>
                      <Text>Cognita (관계 데이터): {integrationConfig.cognita_weight.toFixed(4)}
                        <Text type="secondary"> (F1: 0.2947)</Text>
                      </Text>
                      <Slider
                        min={0}
                        max={1}
                        step={0.0001}
                        value={integrationConfig.cognita_weight}
                        onChange={(value) => setIntegrationConfig(prev => ({...prev, cognita_weight: value}))}
                        disabled={loading || isAnalyzing}
                      />
                    </div>
                    <div>
                      <Text>Chronos (시계열): {integrationConfig.chronos_weight.toFixed(4)}
                        <Text type="secondary"> (F1: 0.7846)</Text>
                      </Text>
                      <Slider
                        min={0}
                        max={1}
                        step={0.0001}
                        value={integrationConfig.chronos_weight}
                        onChange={(value) => setIntegrationConfig(prev => ({...prev, chronos_weight: value}))}
                        disabled={loading || isAnalyzing}
                      />
                    </div>
                    <div>
                      <Text>Sentio (텍스트): {integrationConfig.sentio_weight.toFixed(4)}
                        <Text type="secondary"> (F1: 0.2892)</Text>
                      </Text>
                      <Slider
                        min={0}
                        max={1}
                        step={0.0001}
                        value={integrationConfig.sentio_weight}
                        onChange={(value) => setIntegrationConfig(prev => ({...prev, sentio_weight: value}))}
                        disabled={loading || isAnalyzing}
                      />
                    </div>
                    <div>
                      <Text>Agora (시장): {integrationConfig.agora_weight.toFixed(4)}
                        <Text type="secondary"> (F1: 0.2907)</Text>
                      </Text>
                      <Slider
                        min={0}
                        max={1}
                        step={0.0001}
                        value={integrationConfig.agora_weight}
                        onChange={(value) => setIntegrationConfig(prev => ({...prev, agora_weight: value}))}
                        disabled={loading || isAnalyzing}
                      />
                    </div>

                    <div style={{ marginTop: '16px' }}>
                      <Text strong>통합 위험도 임계치:</Text>
                      <div style={{ marginTop: '8px' }}>
                        <Text>고위험 임계치: {integrationConfig.high_risk_threshold}</Text>
                        <Slider
                          min={0.5}
                          max={1}
                          step={0.05}
                          value={integrationConfig.high_risk_threshold}
                          onChange={(value) => setIntegrationConfig(prev => ({...prev, high_risk_threshold: value}))}
                          disabled={loading || isAnalyzing}
                          marks={{
                            0.5: '0.5',
                            0.7: '0.7',
                            0.9: '0.9',
                            1.0: '1.0'
                          }}
                        />
                      </div>
                      <div style={{ marginTop: '8px' }}>
                        <Text>중위험 임계치: {integrationConfig.medium_risk_threshold}</Text>
                        <Slider
                          min={0.2}
                          max={0.8}
                          step={0.05}
                          value={integrationConfig.medium_risk_threshold}
                          onChange={(value) => setIntegrationConfig(prev => ({...prev, medium_risk_threshold: value}))}
                          disabled={loading || isAnalyzing}
                          marks={{
                            0.2: '0.2',
                            0.4: '0.4',
                            0.6: '0.6',
                            0.8: '0.8'
                          }}
                        />
                      </div>
                    </div>
                    
                    <Alert
                      message={`가중치 합계: ${(
                        integrationConfig.structura_weight + 
                        integrationConfig.cognita_weight + 
                        integrationConfig.chronos_weight + 
                        integrationConfig.sentio_weight + 
                        integrationConfig.agora_weight
                      ).toFixed(4)}`}
                      type={Math.abs((
                        integrationConfig.structura_weight + 
                        integrationConfig.cognita_weight + 
                        integrationConfig.chronos_weight + 
                        integrationConfig.sentio_weight + 
                        integrationConfig.agora_weight
                      ) - 1.0) < 0.01 ? "success" : "warning"}
                      showIcon
                    />
                  </Space>
                </Col>
              </Row>
            </Space>
          </Card>
        </Col>

        {/* 분석 실행 섹션 */}
        <Col span={24}>
          <Card title="7단계: 통합 배치 분석 실행" extra={<ApiOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                type="primary"
                size="large"
                icon={<ApiOutlined />}
                onClick={runBatchAnalysis}
                disabled={!agentFiles.structura || isAnalyzing}
                loading={isAnalyzing}
              >
                {isAnalyzing ? '분석 진행 중...' : '배치 분석 시작'}
              </Button>

              {isAnalyzing && (
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>전체 진행률</Text>
                    <Progress
                      percent={analysisProgress.overall}
                      status="active"
                      strokeColor={{
                        '0%': '#108ee9',
                        '100%': '#87d068',
                      }}
                    />
                  </div>
                  
                  <Row gutter={[16, 8]}>
                    <Col span={12}>
                      <Text>Structura (HR 분석)</Text>
                      <Progress 
                        percent={analysisProgress.structura} 
                        size="small"
                        strokeColor="#1890ff"
                      />
                    </Col>
                    <Col span={12}>
                      <Text>Cognita (관계 분석)</Text>
                      <Progress 
                        percent={analysisProgress.cognita} 
                        size="small"
                        strokeColor="#52c41a"
                      />
                    </Col>
                    <Col span={12}>
                      <Text>Chronos (시계열 분석)</Text>
                      <Progress 
                        percent={analysisProgress.chronos} 
                        size="small"
                        strokeColor="#fa8c16"
                      />
                    </Col>
                    <Col span={12}>
                      <Text>Sentio (감정 분석)</Text>
                      <Progress 
                        percent={analysisProgress.sentio} 
                        size="small"
                        strokeColor="#eb2f96"
                      />
                    </Col>
                    <Col span={24}>
                      <Text>Agora (시장 분석)</Text>
                      <Progress 
                        percent={analysisProgress.agora} 
                        size="small"
                        strokeColor="#722ed1"
                      />
                    </Col>
                  </Row>
                </Space>
              )}

              <Text type="secondary">
                Supervisor가 각 에이전트를 순차적으로 실행하여 종합적인 분석을 수행합니다.
              </Text>
            </Space>
          </Card>
        </Col>

        {/* 분석 결과 섹션 */}
        {analysisResults && (
          <Col span={24}>
            <Card title="3단계: 분석 결과" extra={<BarChartOutlined />}>
              <Space direction="vertical" style={{ width: '100%' }}>
                {/* 요약 통계 */}
                <Row gutter={16}>
                  <Col span={6}>
                    <Statistic
                      title="총 직원 수"
                      value={analysisResults.total_employees}
                      prefix={<CheckCircleOutlined />}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="고위험군"
                      value={analysisResults.results?.filter(r => {
                        const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                        return score && calculateRiskLevel(score) === 'HIGH';
                      }).length || 0}
                      valueStyle={{ color: '#cf1322' }}
                      prefix={<ExclamationCircleOutlined />}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="중위험군"
                      value={analysisResults.results?.filter(r => {
                        const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                        return score && calculateRiskLevel(score) === 'MEDIUM';
                      }).length || 0}
                      valueStyle={{ color: '#fa8c16' }}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="저위험군"
                      value={analysisResults.results?.filter(r => {
                        const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                        return score && calculateRiskLevel(score) === 'LOW';
                      }).length || 0}
                      valueStyle={{ color: '#52c41a' }}
                    />
                  </Col>
                </Row>

                {/* 결과 액션 버튼들 */}
                <Row gutter={16} style={{ marginBottom: 16 }}>
                  <Col>
                    <Button 
                      type="primary" 
                      icon={<DownloadOutlined />}
                      onClick={() => exportResults('excel')}
                    >
                      Excel로 내보내기
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<FileTextOutlined />}
                      onClick={() => exportResults('pdf')}
                    >
                      PDF 보고서
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<DashboardOutlined />}
                      onClick={() => navigateToSystemStatus()}
                    >
                      시스템 현황 보기
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<BarChartOutlined />}
                      onClick={() => navigateToVisualization()}
                    >
                      시각화 대시보드
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<TeamOutlined />}
                      onClick={() => navigateToRelationshipAnalysis()}
                    >
                      관계 분석
                    </Button>
                  </Col>
                </Row>

                {/* 결과 테이블 */}
                <Table
                  columns={resultColumns}
                  dataSource={analysisResults.results}
                  rowKey="employee_number"
                  pagination={{
                    pageSize: 10,
                    showSizeChanger: true,
                    showQuickJumper: true,
                    showTotal: (total, range) =>
                      `${range[0]}-${range[1]} of ${total} 직원`
                  }}
                  scroll={{ x: 800 }}
                />

                {/* Integration 보고서 섹션 */}
                {analysisResults.integration_report && (
                  <Card title="Integration 종합 보고서" extra={<FileTextOutlined />} style={{ marginTop: 16 }}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      {/* 경영진 요약 */}
                      <Card size="small" title="경영진 요약">
                        <Paragraph>
                          <Text strong>개요:</Text> {analysisResults.integration_report.executive_summary?.overview}
                        </Paragraph>
                        <Paragraph>
                          <Text strong>주요 지표:</Text> {analysisResults.integration_report.executive_summary?.key_metrics}
                        </Paragraph>
                        <Paragraph>
                          <Text strong>긴급도 평가:</Text> {analysisResults.integration_report.executive_summary?.urgency_assessment}
                        </Paragraph>
                      </Card>

                      {/* 주요 발견사항 */}
                      <Card size="small" title="주요 발견사항">
                        <ul>
                          {analysisResults.integration_report.key_findings?.map((finding, index) => (
                            <li key={index}>{finding}</li>
                          ))}
                        </ul>
                      </Card>

                      {/* 권장사항 */}
                      {analysisResults.integration_report.recommendations && (
                        <Card size="small" title="권장사항">
                          <Row gutter={16}>
                            <Col span={8}>
                              <Text strong>즉시 조치사항:</Text>
                              <ul>
                                {analysisResults.integration_report.recommendations.immediate_actions?.map((action, index) => (
                                  <li key={index}>{action}</li>
                                ))}
                              </ul>
                            </Col>
                            <Col span={8}>
                              <Text strong>단기 전략:</Text>
                              <ul>
                                {analysisResults.integration_report.recommendations.short_term_strategies?.map((strategy, index) => (
                                  <li key={index}>{strategy}</li>
                                ))}
                              </ul>
                            </Col>
                            <Col span={8}>
                              <Text strong>장기 이니셔티브:</Text>
                              <ul>
                                {analysisResults.integration_report.recommendations.long_term_initiatives?.map((initiative, index) => (
                                  <li key={index}>{initiative}</li>
                                ))}
                              </ul>
                            </Col>
                          </Row>
                        </Card>
                      )}

                      {/* 워크플로우 효과성 */}
                      {analysisResults.integration_report.detailed_risk_analysis?.workflow_effectiveness && (
                        <Card size="small" title="워크플로우 효과성">
                          <Row gutter={16}>
                            <Col span={8}>
                              <Alert
                                message="순차적 실행"
                                description={analysisResults.integration_report.detailed_risk_analysis.workflow_effectiveness.sequential_execution}
                                type="success"
                                showIcon
                              />
                            </Col>
                            <Col span={8}>
                              <Alert
                                message="데이터 통합"
                                description={analysisResults.integration_report.detailed_risk_analysis.workflow_effectiveness.data_integration}
                                type="success"
                                showIcon
                              />
                            </Col>
                            <Col span={8}>
                              <Alert
                                message="포괄적 분석"
                                description={analysisResults.integration_report.detailed_risk_analysis.workflow_effectiveness.comprehensive_coverage}
                                type="success"
                                showIcon
                              />
                            </Col>
                          </Row>
                        </Card>
                      )}
                    </Space>
                  </Card>
                )}
              </Space>
            </Card>
          </Col>
        )}
      </Row>

    </div>
  );
};

export default BatchAnalysis;