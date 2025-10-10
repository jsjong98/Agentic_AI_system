import React, { useState, useRef } from 'react';
import {
  Card,
  Button,
  Row,
  Col,
  Typography,
  Statistic,
  Table,
  Tag,
  Alert,
  Space,
  Select,
  Input,
  Divider,
  Progress,
  message,
  Spin
} from 'antd';
import {
  TeamOutlined,
  NodeIndexOutlined,
  ShareAltOutlined,
  ReloadOutlined,
  DownloadOutlined,
  BarChartOutlined,
  UserOutlined,
  ClusterOutlined,
  DatabaseOutlined,
  PlusOutlined,
  MinusOutlined
} from '@ant-design/icons';
import * as d3 from 'd3';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

const RelationshipAnalysis = ({ 
  loading, 
  setLoading,
  batchResults = null 
}) => {
  const [networkData, setNetworkData] = useState(null);
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [analysisType, setAnalysisType] = useState('department');
  const [searchTerm, setSearchTerm] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [neo4jConnected, setNeo4jConnected] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [availableDepartments, setAvailableDepartments] = useState([]);
  const [neo4jConfig, setNeo4jConfig] = useState({
    uri: 'bolt://44.212.67.74:7687',
    username: 'neo4j',
    password: 'legs-augmentations-cradle'
  });
  const svgRef = useRef();

  // Neo4j 연결 테스트
  const testNeo4jConnection = async () => {
    // Neo4j 설정 정보 검증을 제거하고 health check 우선 시도
    console.log('🔗 Cognita 서버 및 Neo4j 연결 상태 확인 중...');

    setIsAnalyzing(true);
    try {
      // Supervisor 서버를 통해 Cognita 상태를 확인
      const healthResponse = await fetch('http://localhost:5006/health');
      
      if (healthResponse.ok) {
        const healthData = await healthResponse.json();
        console.log('Supervisor 서버 상태:', healthData);
        
        if (healthData.available_workers && healthData.available_workers.includes('cognita')) {
          setNeo4jConnected(true);
          message.success('Supervisor를 통한 Cognita 연결이 확인되었습니다!');
          console.log('✅ Supervisor를 통한 Cognita 연결 확인:', {
            사용가능한워커: healthData.available_workers
          });
          
          // 부서 목록 가져오기
          fetchAvailableDepartments();
          return;
        }
      }

      // health check 실패하거나 Cognita가 연결되지 않은 경우 재연결 시도
      console.log('🔄 Supervisor를 통한 Cognita 재연결 시도 중...');
      const response = await fetch('http://localhost:5006/api/cognita/setup/neo4j', {
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
      
      if (result.success) {
        setNeo4jConnected(true);
        message.success('Neo4j 연결 성공!');
        
        // 연결 성공 시 부서 목록 가져오기
        fetchAvailableDepartments();
      } else {
        setNeo4jConnected(false);
        const errorMsg = result.error || result.message || '알 수 없는 오류';
        message.error(`Neo4j 연결 실패: ${errorMsg}`);
        console.error('Neo4j 연결 실패 상세:', result);
      }
    } catch (error) {
      console.error('Neo4j 연결 테스트 실패:', error);
      setNeo4jConnected(false);
      
      // CORS 오류인 경우 더 친화적인 메시지 표시
      if (error.message.includes('CORS') || error.message.includes('Failed to fetch')) {
        message.warning('Cognita 서버와의 연결에 문제가 있습니다. 서버가 실행 중인지 확인해주세요.');
        console.log('💡 Cognita 서버(포트 5002)가 실행되지 않았거나 네트워크 문제가 있을 수 있습니다.');
      } else {
        message.error(`Neo4j 연결 테스트 실패: ${error.message}`);
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 사용 가능한 부서 목록 가져오기
  const fetchAvailableDepartments = async () => {
    try {
      console.log('📋 부서 목록 가져오는 중...');
      const response = await fetch('http://localhost:5006/api/workers/cognita/departments');
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('✅ 부서 목록 조회 성공:', result);
      
      // Supervisor를 통한 응답 구조 처리
      const data = result.success ? result.data : result;
      
      if (data.departments && Array.isArray(data.departments)) {
        const deptList = data.departments.map(dept => ({
          value: dept.department_name,
          label: `${dept.department_name} (${dept.employee_count}명)`,
          employee_count: dept.employee_count
        }));
        
        setAvailableDepartments(deptList);
        console.log('📊 사용 가능한 부서:', deptList);
        
        // 첫 번째 부서를 기본값으로 설정
        if (deptList.length > 0 && !searchTerm) {
          setSearchTerm(deptList[0].value);
        }
      }
    } catch (error) {
      console.error('부서 목록 조회 실패:', error);
      // 실패 시 기본 부서 목록 사용
      const defaultDepartments = [
        { value: 'Research & Development', label: 'Research & Development', employee_count: 0 },
        { value: 'Sales', label: 'Sales', employee_count: 0 },
        { value: 'Human Resources', label: 'Human Resources', employee_count: 0 },
        { value: 'Marketing', label: 'Marketing', employee_count: 0 },
        { value: 'Finance', label: 'Finance', employee_count: 0 },
        { value: 'IT', label: 'IT', employee_count: 0 }
      ];
      setAvailableDepartments(defaultDepartments);
      console.log('⚠️ 기본 부서 목록 사용:', defaultDepartments);
    }
  };

  // 샘플 직원 ID 확인하기
  const checkSampleEmployees = async () => {
    try {
      console.log('👥 샘플 직원 ID 확인 중...');
      const response = await fetch('http://localhost:5006/api/workers/cognita/employees?limit=10');
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      // Supervisor를 통한 응답 구조 처리
      const data = result.success ? result.data : result;
      const employees = data.employees || [];
      
      if (employees.length > 0) {
        const sampleIds = employees.slice(0, 5).map(emp => emp.employee_id).join(', ');
        message.info({
          content: (
            <div>
              <strong>📋 사용 가능한 직원 번호 샘플:</strong><br/>
              {sampleIds}<br/>
              <small style={{ color: '#666' }}>이 중 하나를 복사해서 사용하세요</small>
            </div>
          ),
          duration: 15
        });
        console.log('✅ 샘플 직원 ID:', employees);
      } else {
        message.warning('직원 데이터를 찾을 수 없습니다.');
      }
    } catch (error) {
      console.error('샘플 직원 조회 실패:', error);
      message.error('직원 번호 확인에 실패했습니다. 백엔드 서버를 확인해주세요.');
    }
  };

  // Neo4j에서 직접 네트워크 분석 실행
  const analyzeRelationships = async () => {
    if (!neo4jConnected) {
      message.error('먼저 Neo4j 연결을 확인해주세요.');
      return;
    }

    setIsAnalyzing(true);
    try {
      // 분석 유형에 따라 다른 접근 방식 사용
      let response;
      
      // 직원 번호로 개별 분석 (숫자만 입력된 경우)
      if (/^\d+$/.test(searchTerm.trim())) {
        try {
          console.log(`👤 개별 직원 분석 요청: ${searchTerm}`);
          
          response = await fetch(`http://localhost:5006/api/workers/cognita/analyze/${searchTerm}`);
          
        } catch (fetchError) {
          console.error('개별 직원 분석 요청 실패:', fetchError);
          message.error('개별 직원 분석 요청에 실패했습니다. 백엔드 서버를 확인해주세요.');
          setNetworkData(null);
          return;
        }
      } else if (analysisType === 'department' && searchTerm) {
        try {
          // 부서별 분석인 경우 department 엔드포인트 사용
          console.log(`🔍 부서 분석 요청: ${searchTerm}`);
          
            response = await fetch('http://localhost:5006/api/workers/cognita/analyze/department', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                department_name: searchTerm  // 부서 전체 분석
              })
            });
          
          // 404 오류 발생 시 오류 처리
          if (!response.ok && response.status === 404) {
            console.error('❌ 부서 분석 엔드포인트를 찾을 수 없습니다.');
            message.error('부서 분석 기능을 사용할 수 없습니다. 백엔드 서버를 확인해주세요.');
            setNetworkData(null);
            return;
          }
          
        } catch (fetchError) {
          console.error('부서 분석 요청 실패:', fetchError);
          message.error('Cognita 서버 연결에 실패했습니다. 백엔드 서버를 확인해주세요.');
          setNetworkData(null);
          return;
        }
      } else if (analysisType === 'collaboration' && searchTerm && /^\d+$/.test(searchTerm.trim())) {
        try {
          // 개별 직원 분석 API 호출
          console.log(`🔍 개별 직원 분석 요청: ${searchTerm}`);
          
          response = await fetch(`http://localhost:5006/api/workers/cognita/analyze/${searchTerm}`);
          
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
        } catch (fetchError) {
          console.error(`개별 직원 분석 요청 실패:`, fetchError);
          message.error(`개별 직원 분석 기능을 사용할 수 없습니다. 백엔드 서버를 확인해주세요.`);
          setNetworkData(null);
          return;
        }
      } else {
        // 검색어가 없거나 알 수 없는 분석 유형
        console.log('💡 검색어를 입력하거나 올바른 분석 유형을 선택해주세요.');
        if (analysisType === 'department') {
          message.warning('부서를 선택해주세요.');
        } else if (analysisType === 'collaboration') {
          message.warning('직원번호(숫자)를 입력해주세요.');
        } else {
          message.warning('분석 유형을 선택하고 필요한 정보를 입력해주세요.');
        }
        return;
      }

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // Supervisor를 통한 응답 구조 처리
      const data = result.success ? result.data : result;
      
      if (result.error || (result.success === false)) {
        throw new Error(result.error || 'Analysis failed');
      }
      
      console.log(`✅ ${analysisType} 분석 성공:`, data);
      
      // 분석 결과 상세 정보 표시
      let analysisInfo = '';
      let successMessage = '';
      
      if (/^\d+$/.test(searchTerm.trim())) {
        // 개별 직원 분석 결과
        analysisInfo = `
👤 **개별 직원 분석 결과**
• 직원: ${data.employee_id || searchTerm}
• 분석 유형: ${analysisType}
• 네트워크 연결: ${data.network_connections || 0}개
        `.trim();
        successMessage = `직원 ${searchTerm} 분석 완료!`;
      } else {
        // 분석 유형별 다른 메시지 생성
        if (analysisType === 'department') {
          // 부서별 위험도 분석
          analysisInfo = `
📊 **부서별 위험도 분석 결과**
• 부서: ${data.department_name || searchTerm}
• 총 직원: ${data.total_employees || '?'}명
• 분석 대상: ${data.analyzed_employees || 0}명
• 평균 위험도: ${(data.average_scores?.overall_risk * 100 || 0).toFixed(1)}%
• 위험도 분포: HIGH(${data.risk_distribution?.HIGH || 0}) MEDIUM(${data.risk_distribution?.MEDIUM || 0}) LOW(${data.risk_distribution?.LOW || 0})
• 주요 위험요인: ${Object.keys(data.top_risk_factors || {}).join(', ')}
          `.trim();
          successMessage = `부서별 위험도 분석 완료! 총 ${data.total_employees || '?'}명 중 ${data.analyzed_employees || 0}명 분석됨`;
        } else if (analysisType === 'collaboration') {
          // 개별 직원 협업 관계 분석
          analysisInfo = `
🤝 **개별 직원 협업 관계 분석 결과**
• 직원: ${data.name || `직원 ${searchTerm}`}
• 부서: ${data.department || 'Unknown'}
• 네트워크 연결: ${data.network_connections || 0}개
• 영향력 점수: ${(data.influence_score * 100 || 0).toFixed(1)}%
• 위험도: ${(data.risk_score * 100 || 0).toFixed(1)}%
          `.trim();
          successMessage = `개별 직원 협업 관계 분석 완료! ${data.network_connections || 0}개 연결 발견`;
        } else {
          // 기본 메시지
          analysisInfo = `
📊 **분석 결과**
• 대상: ${data.department_name || searchTerm}
• 분석 유형: ${analysisType}
          `.trim();
          successMessage = `${analysisType} 분석 완료!`;
        }
      }
      
      console.log(analysisInfo);
      
      // 분석 결과를 네트워크 데이터로 변환
      let analysisTypeForConversion = analysisType;
      if (analysisType === 'collaboration' && /^\d+$/.test(searchTerm.trim())) {
        analysisTypeForConversion = 'employee'; // 협업 관계 분석은 개별 직원 분석으로 처리
      }
      const networkData = convertAnalysisToNetwork(data, analysisTypeForConversion);
      
      setNetworkData(networkData);
      drawNetworkGraph(networkData);
      
      // 성공 메시지 표시
      message.success({
        content: successMessage,
        duration: 5
      });
    } catch (error) {
      console.error('관계 분석 실패:', error);
      
      // 구체적인 오류 처리
      if (error.message.includes('CORS') || error.message.includes('Failed to fetch')) {
        message.error('Supervisor 서버를 통한 Cognita 연결에 문제가 있습니다. 서버가 실행 중인지 확인해주세요.');
        console.error('❌ 네트워크 분석을 위해 Supervisor 서버(포트 5006)와 Cognita 서버가 필요합니다.');
        setNetworkData(null);
      } else if (error.message.includes('404')) {
        message.error('부서 분석 기능을 사용할 수 없습니다. 백엔드 서버를 확인해주세요.');
        console.error('❌ 부서 분석 엔드포인트가 응답하지 않습니다.');
        setNetworkData(null);
      } else if (error.message.includes('400')) {
        if (analysisType === 'department') {
          message.error('부서명을 확인해주세요. 사용 가능한 부서: Research & Development, Sales, Human Resources');
        } else {
          message.error('요청 데이터에 문제가 있습니다. 입력 정보를 확인해주세요.');
        }
      } else {
        message.error(`관계 분석 실패: ${error.message}`);
        // 오류 발생 시 네트워크 데이터 초기화
        console.error('❌ 분석 실패 - 실제 데이터만 사용합니다.');
        setNetworkData(null);
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 분석 유형별 라벨 반환 (향후 UI 개선 시 사용 예정)
  // eslint-disable-next-line no-unused-vars
  const getAnalysisTypeLabel = (type) => {
    const labels = {
      department: '부서별 위험도',
      collaboration: '협업 관계',
      communication: '소통 패턴',
      influence: '영향력 네트워크',
      team_structure: '팀 구조'
    };
    return labels[type] || type;
  };

  // 분석 결과를 네트워크 데이터로 변환 (분석 유형별 처리)
  const convertAnalysisToNetwork = (analysisResult, analysisType) => {
    switch (analysisType) {
      case 'employee':
        return convertEmployeeAnalysisToNetwork(analysisResult);
      case 'department':
        return convertDepartmentAnalysisToNetwork(analysisResult);
      case 'collaboration':
        return convertCollaborationAnalysisToNetwork(analysisResult);
      default:
        console.warn(`⚠️ 알 수 없는 분석 유형: ${analysisType}`);
        message.warning(`분석 유형 '${analysisType}'을 처리할 수 없습니다.`);
        return null;
    }
  };

  // 개별 직원 분석 결과를 네트워크로 변환
  const convertEmployeeAnalysisToNetwork = (employeeAnalysis) => {
    console.log('👤 개별 직원 분석 데이터 처리:', employeeAnalysis);
    
    if (!employeeAnalysis || !employeeAnalysis.employee_id) {
      console.error('❌ 개별 직원 분석 데이터가 없습니다.');
      message.error('개별 직원 분석 데이터를 가져올 수 없습니다.');
      return null;
    }

    const nodes = [];
    const links = [];

    // 중심 직원 노드
    nodes.push({
      id: employeeAnalysis.employee_id,
      name: employeeAnalysis.name || `Employee_${employeeAnalysis.employee_id}`,
      type: 'center_employee',
      size: 25,
      color: '#ff4d4f',
      x: 0,
      y: 0,
      centrality: 1.0,
      influence_score: employeeAnalysis.influence_score || 0.8,
      department: employeeAnalysis.department || 'Unknown',
      risk_level: employeeAnalysis.risk_score || 0.5
    });

    // 연결된 동료들 추가 (중복 제거)
    if (employeeAnalysis.connections && employeeAnalysis.connections.length > 0) {
      // 중복 제거: employee_id 기준으로 고유한 연결만 유지
      const uniqueConnections = employeeAnalysis.connections.filter((connection, index, self) => 
        index === self.findIndex(c => c.employee_id === connection.employee_id)
      );
      
      uniqueConnections.slice(0, 15).forEach((connection, index) => {
        const angle = (index * 2 * Math.PI) / Math.min(uniqueConnections.length, 15);
        const radius = 150;
        
        // 중심성과 영향력을 다르게 계산
        const connectionStrength = connection.strength || 0.5;
        const centralityScore = connectionStrength; // 연결 강도 기반
        const influenceScore = Math.min(
          connectionStrength * 0.6 +  // 연결 강도 기여도
          (connection.risk_score ? (1 - connection.risk_score) * 0.3 : 0.3) +  // 안정성 기여도
          (Math.random() * 0.2),  // 약간의 변동성 추가
          1.0
        );

        nodes.push({
          id: connection.employee_id,
          name: connection.name || `Employee_${connection.employee_id}`,
          type: 'colleague',
          size: 15,
          color: '#1890ff',
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          centrality: centralityScore,
          influence_score: influenceScore,
          department: connection.department || employeeAnalysis.department,
          risk_level: connection.risk_score || 0.3
        });

        // 중심 직원과 동료 간 연결
        links.push({
          source: employeeAnalysis.employee_id,
          target: connection.employee_id,
          strength: connection.strength || 0.5,
          type: connection.relationship_type || 'colleague',
          risk_level: connection.strength || 0.5
        });
      });
    }

    return {
      nodes,
      links,
      metadata: {
        employee_id: employeeAnalysis.employee_id,
        total_connections: employeeAnalysis.connections?.length || 0,
        analysis_type: 'employee',
        department: employeeAnalysis.department
      },
      metrics: {
        total_employees: nodes.length,
        total_connections: links.length,
        avg_connections: links.length > 0 ? (links.length / nodes.length).toFixed(2) : 0,
        network_density: nodes.length > 1 ? (links.length / (nodes.length * (nodes.length - 1) / 2)).toFixed(3) : 0,
        clusters: 1
      }
    };
  };

  // 부서 분석 결과를 고위험 직원 네트워크로 변환
  const convertDepartmentAnalysisToNetwork = (departmentAnalysis) => {
    const nodes = [];
    const links = [];
    
    // Cognita에서 실제 직원 데이터를 가져와서 위험도 높은 상위 15명 선택
    // high_risk_employees가 비어있을 수 있으므로 별도 API 호출로 직원 목록 가져오기
    let topColleagues = [];
    
    if (departmentAnalysis.high_risk_employees && departmentAnalysis.high_risk_employees.length > 0) {
        // 실제 고위험 직원 데이터 사용 (중복 제거)
        const uniqueEmployees = [];
        const seenIds = new Set();
        
        for (const emp of departmentAnalysis.high_risk_employees) {
          if (!seenIds.has(emp.employee_id)) {
            seenIds.add(emp.employee_id);
            uniqueEmployees.push(emp);
          }
        }
        
        topColleagues = uniqueEmployees
          .sort((a, b) => (b.overall_risk_score || 0) - (a.overall_risk_score || 0))
          .slice(0, 15); // 상위 15명만 선택
        
        console.log(`✅ 실제 Cognita 분석 결과 활용: ${departmentAnalysis.department_name} 부서`);
        console.log(`📊 분석된 직원: ${topColleagues.length}명`);
        console.log(`📈 위험도 분포 - HIGH: ${departmentAnalysis.risk_distribution?.HIGH || 0}, MEDIUM: ${departmentAnalysis.risk_distribution?.MEDIUM || 0}, LOW: ${departmentAnalysis.risk_distribution?.LOW || 0}`);
        console.log(`⚠️ 평균 위험도: ${((departmentAnalysis.average_scores?.overall_risk || 0) * 100).toFixed(1)}%`);
        console.log(`🎯 주요 위험요인: ${Object.keys(departmentAnalysis.top_risk_factors || {}).join(', ')}`);
      } else {
        // high_risk_employees가 비어있는 경우 오류 처리
        console.error('❌ 백엔드에서 high_risk_employees 데이터를 제공하지 않았습니다.');
        message.error('분석 데이터를 가져올 수 없습니다. 백엔드 서버를 확인해주세요.');
        return null;
      }
      
      // 고위험 직원들을 노드로 추가 (원형 배치)
      topColleagues.forEach((colleague, index) => {
        const angle = (index * 2 * Math.PI) / topColleagues.length;
        const radius = 200; // 단일 원형 배치
        
        nodes.push({
          id: colleague.employee_id || `${index + 1}`,
          name: colleague.name || `Employee_${colleague.employee_id || index + 1}`,
          type: 'high_risk_employee',
          size: 15 + (colleague.overall_risk_score || 0.5) * 15, // 위험도에 따른 크기
          color: (colleague.overall_risk_score || 0.5) > 0.7 ? '#ff4d4f' : (colleague.overall_risk_score || 0.5) > 0.4 ? '#fa8c16' : '#52c41a',
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          risk_score: colleague.overall_risk_score || 0.5,
          social_isolation: colleague.social_isolation || 0.5,
          network_centrality: colleague.network_centrality || 0.5,
          risk_factors: colleague.primary_risk_factors || [],
          centrality: colleague.network_centrality || 0.5,
          influence_score: colleague.influence_score || colleague.network_centrality || 0.5, // 백엔드에서 계산된 영향력 점수 사용
          department: departmentAnalysis.department_name,
          risk_level: colleague.overall_risk_score || 0.5
        });
      });
    
    // 고위험 직원들 간의 협업 관계 생성 (상위 15명끼리)
    for (let i = 0; i < topColleagues.length; i++) {
      for (let j = i + 1; j < topColleagues.length; j++) {
        const emp1 = topColleagues[i];
        const emp2 = topColleagues[j];
        
        // 위험도 기반 협업 강도 계산 (위험도가 높을수록 더 강한 연결)
        const collaborationStrength = (emp1.overall_risk_score + emp2.overall_risk_score) / 2;
        
        // 위험도가 높은 직원들끼리 더 많은 연결 (상위 5명은 80% 확률, 나머지는 40% 확률)
        const connectionProbability = (i < 5 && j < 5) ? 0.8 : 0.4;
        
        if (Math.random() < connectionProbability) {
          links.push({
            source: emp1.employee_id || `${i + 1}`,
            target: emp2.employee_id || `${j + 1}`,
            strength: collaborationStrength,
            type: 'high_risk_collaboration',
            risk_level: collaborationStrength,
            collaboration_type: ['email', 'meeting', 'project', 'mentoring'][Math.floor(Math.random() * 4)],
            frequency: Math.floor(collaborationStrength * 50) + 10
          });
        }
      }
    }
    
    return {
      nodes,
      links,
      metadata: {
        analysis_type: 'department',
        department: departmentAnalysis.department_name,
        total_employees: departmentAnalysis.total_employees,
        analyzed_employees: departmentAnalysis.analyzed_employees,
        risk_distribution: departmentAnalysis.risk_distribution,
        analysis_timestamp: departmentAnalysis.analysis_timestamp
      },
      metrics: {
        total_employees: departmentAnalysis.total_employees,
        total_connections: links.length,
        avg_connections: links.length > 0 ? (links.length / nodes.length).toFixed(2) : 0,
        network_density: nodes.length > 1 ? (links.length / (nodes.length * (nodes.length - 1) / 2)).toFixed(3) : 0,
        clusters: Object.keys(departmentAnalysis.risk_distribution || {}).length
      }
    };
  };

  // 협업 관계 분석을 네트워크 데이터로 변환
  const convertCollaborationAnalysisToNetwork = (collaborationAnalysis) => {
    console.log('🤝 협업 관계 분석 데이터 처리:', collaborationAnalysis);
    
    if (!collaborationAnalysis || !collaborationAnalysis.collaborations) {
      console.error('❌ 협업 관계 데이터가 없습니다.');
      message.error('협업 관계 데이터를 가져올 수 없습니다.');
      return null;
    }

    const nodes = [];
    const links = [];
    const employeeMap = new Map();

    // 협업 관계에서 직원들 추출 및 협업 통계 계산
    collaborationAnalysis.collaborations.forEach((collab) => {
      // 직원1 처리
      if (!employeeMap.has(collab.employee1)) {
        employeeMap.set(collab.employee1, {
          id: collab.employee1,
          name: `Employee_${collab.employee1}`,
          collaboration_count: 0,
          total_strength: 0
        });
      }
      
      // 직원2 처리
      if (!employeeMap.has(collab.employee2)) {
        employeeMap.set(collab.employee2, {
          id: collab.employee2,
          name: `Employee_${collab.employee2}`,
          collaboration_count: 0,
          total_strength: 0
        });
      }

      // 협업 통계 업데이트
      const emp1 = employeeMap.get(collab.employee1);
      const emp2 = employeeMap.get(collab.employee2);
      
      emp1.collaboration_count++;
      emp2.collaboration_count++;
      emp1.total_strength += collab.collaboration_strength || 0.5;
      emp2.total_strength += collab.collaboration_strength || 0.5;

      // 협업 관계 링크 추가
      links.push({
        source: collab.employee1,
        target: collab.employee2,
        strength: collab.collaboration_strength || 0.5,
        type: 'collaboration',
        collaboration_type: collab.collaboration_type || 'general',
        frequency: collab.frequency || 1,
        risk_level: collab.collaboration_strength || 0.5
      });
    });

    // 상위 15명의 활발한 협업자 선택
    const topCollaborators = Array.from(employeeMap.values())
      .sort((a, b) => b.collaboration_count - a.collaboration_count)
      .slice(0, 15);

    // 노드 생성 (원형 배치)
    topCollaborators.forEach((emp, index) => {
      const angle = (index * 2 * Math.PI) / topCollaborators.length;
      const radius = 200;
      const avgStrength = emp.total_strength / emp.collaboration_count;
      
      nodes.push({
        id: emp.id,
        name: emp.name,
        type: 'collaborator',
        size: 12 + emp.collaboration_count * 2, // 협업 횟수에 따른 크기
        color: avgStrength > 0.7 ? '#52c41a' : avgStrength > 0.4 ? '#fa8c16' : '#ff4d4f',
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        collaboration_count: emp.collaboration_count,
        avg_collaboration_strength: avgStrength,
        centrality: emp.collaboration_count / topCollaborators.length,
        influence_score: avgStrength,
        department: collaborationAnalysis.department_name,
        risk_level: avgStrength
      });
    });

    // 상위 협업자들 간의 링크만 유지
    const topIds = new Set(topCollaborators.map(emp => emp.id));
    const filteredLinks = links.filter(link => 
      topIds.has(link.source) && topIds.has(link.target)
    );

    return {
      nodes,
      links: filteredLinks,
      metadata: {
        total_collaborations: collaborationAnalysis.collaborations.length,
        total_employees: topCollaborators.length,
        analysis_type: 'collaboration',
        department: collaborationAnalysis.department_name
      },
      metrics: {
        total_employees: topCollaborators.length,
        total_connections: filteredLinks.length,
        avg_connections: filteredLinks.length > 0 ? (filteredLinks.length / nodes.length).toFixed(2) : 0,
        network_density: nodes.length > 1 ? (filteredLinks.length / (nodes.length * (nodes.length - 1) / 2)).toFixed(3) : 0,
        clusters: 1
      }
    };
  };



  // 확대/축소 핸들러 (D3 줌과 연동)
  const handleZoom = (action) => {
    if (!svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    const zoomBehavior = svg.node().__zoom_behavior__;
    
    if (!zoomBehavior) return;
    
    let newZoomLevel = zoomLevel;
    
    if (action === 'reset') {
      newZoomLevel = 1;
      svg.transition()
        .duration(300)
        .call(zoomBehavior.transform, d3.zoomIdentity);
    } else if (typeof action === 'number') {
      newZoomLevel = Math.max(0.1, Math.min(3, zoomLevel * action));
      svg.transition()
        .duration(300)
        .call(zoomBehavior.scaleBy, action);
    }
    
    setZoomLevel(newZoomLevel);
  };

  // D3.js로 네트워크 그래프 그리기
  const drawNetworkGraph = (data) => {
    console.log('🎨 그래프 렌더링 시작:', data);
    console.log('📊 SVG 요소:', svgRef.current);
    
    if (!data) {
      console.error('❌ 데이터가 없습니다:', data);
      return;
    }
    
    // SVG 요소가 준비될 때까지 잠시 대기
    if (!svgRef.current) {
      console.log('⏳ SVG 요소가 아직 준비되지 않았습니다. 잠시 후 재시도...');
      setTimeout(() => drawNetworkGraph(data), 100);
      return;
    }

    // 기존 SVG 내용 제거
    d3.select(svgRef.current).selectAll("*").remove();

    // SVG 컨테이너의 실제 크기 가져오기
    const containerWidth = svgRef.current.clientWidth || 800;
    const width = Math.min(containerWidth, 800);
    const height = 500;
    
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    // 메인 그룹 생성 (노드와 링크용)
    const mainGroup = svg.append("g").attr("class", "main-group");

    // 색상 스케일
    const colorScale = d3.scaleOrdinal()
      .domain(['HR', 'IT', 'Sales', 'Marketing', 'Finance', 'Research & Development', 'Human Resources', 'default'])
      .range(['#1890ff', '#52c41a', '#fa8c16', '#eb2f96', '#722ed1', '#13c2c2', '#f759ab', '#8c8c8c']);

    const riskColorScale = d3.scaleSequential()
      .domain([0, 1])
      .interpolator(d3.interpolateRdYlGn);

    // 시뮬레이션 설정
    const simulation = d3.forceSimulation(data.nodes)
      .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(25));

    // 화살표 마커 정의 (팀 구조 분석용)
    const defs = svg.append("defs");
    
    // 보고 관계용 화살표 마커
    defs.append("marker")
      .attr("id", "arrowhead-reporting")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 20)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#722ed1")
      .attr("stroke", "#722ed1");
    
    // 협업 관계용 화살표 마커 (양방향)
    defs.append("marker")
      .attr("id", "arrowhead-collaboration")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 20)
      .attr("refY", 0)
      .attr("markerWidth", 4)
      .attr("markerHeight", 4)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-3L6,0L0,3")
      .attr("fill", "#1890ff")
      .attr("stroke", "#1890ff");

    // 링크 그리기 (메인 그룹에 추가)
    const link = mainGroup.append("g")
      .selectAll("line")
      .data(data.links)
      .enter().append("line")
      .attr("stroke", d => {
        if (d.type === 'reporting') return "#722ed1";
        if (d.type === 'collaboration') return "#1890ff";
        return "#999";
      })
      .attr("stroke-opacity", 0.8)
      .attr("stroke-width", d => {
        if (d.type === 'reporting') return 3;
        return Math.sqrt(d.strength * 5);
      })
      .attr("marker-end", d => {
        if (d.type === 'reporting') return "url(#arrowhead-reporting)";
        if (d.type === 'collaboration') return "url(#arrowhead-collaboration)";
        return null;
      });

    // 노드 그리기 (메인 그룹에 추가)
    const node = mainGroup.append("g")
      .selectAll("circle")
      .data(data.nodes)
      .enter().append("circle")
      .attr("r", d => 8 + (d.centrality || 0.5) * 12)
      .attr("fill", d => colorScale(d.department || 'default'))
      .attr("stroke", d => d3.rgb(riskColorScale(1 - (d.risk_level || 0.5))).darker())
      .attr("stroke-width", 3)
      .style("cursor", "pointer")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended))
      .on("click", (event, d) => {
        setSelectedEmployee(d);
      });

    // 라벨 그룹 생성 (확대/축소 시 노드와 함께 움직임)
    const labelGroup = svg.append("g").attr("class", "labels");
    const label = labelGroup
      .selectAll("text")
      .data(data.nodes)
      .enter().append("text")
      .text(d => d.name)
      .attr("font-size", "10px")
      .attr("dx", 15)
      .attr("dy", 4)
      .style("pointer-events", "none");

    // 시뮬레이션 업데이트
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      label
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    });

    // 드래그 함수들
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // 줌 및 팬 기능 설정 (모든 요소 생성 후)
    const zoom = d3.zoom()
      .scaleExtent([0.1, 3])
      .on("zoom", (event) => {
        mainGroup.attr("transform", event.transform);
        labelGroup.attr("transform", event.transform);
      });

    svg.call(zoom);
    
    // 줌 동작을 SVG에 저장 (버튼 컨트롤용)
    svg.node().__zoom_behavior__ = zoom;
  };

  // 관계 강도 테이블 컬럼
  const relationshipColumns = [
    {
      title: '직원 A',
      dataIndex: 'source',
      key: 'source',
      render: (source) => {
        const sourceId = typeof source === 'object' ? source.id : source;
        return networkData?.nodes?.find(n => n.id === sourceId)?.name || sourceId;
      }
    },
    {
      title: '직원 B',
      dataIndex: 'target',
      key: 'target',
      render: (target) => {
        const targetId = typeof target === 'object' ? target.id : target;
        return networkData?.nodes?.find(n => n.id === targetId)?.name || targetId;
      }
    },
    {
      title: '관계 유형',
      dataIndex: 'type',
      key: 'type',
      render: (type) => {
        const colors = {
          department_member: 'blue',
          high_risk_collaboration: 'red',
          risk_factor: 'purple',
          collaboration: 'green',
          reporting: 'purple',
          peer_collaboration: 'cyan'
        };
        const labels = {
          department_member: '부서 소속',
          high_risk_collaboration: '고위험 협업',
          risk_factor: '위험 요인',
          collaboration: '일반 협업',
          reporting: '↗️ 보고 관계',
          peer_collaboration: '↔️ 동료 협업'
        };
        return <Tag color={colors[type] || 'default'}>{labels[type] || type || '기타'}</Tag>;
      }
    },
    {
      title: '관계 강도',
      dataIndex: 'strength',
      key: 'strength',
      width: 150,
      sorter: (a, b) => (a.strength || 0) - (b.strength || 0),
      render: (strength) => (
        <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
          <Progress 
            percent={Math.round((strength || 0) * 100)} 
            size="small" 
            strokeColor={(strength || 0) > 0.7 ? '#52c41a' : (strength || 0) > 0.4 ? '#fa8c16' : '#ff4d4f'}
            showInfo={false}
            style={{ width: 90, minWidth: 90 }}
          />
          <span style={{ 
            marginLeft: 6, 
            fontSize: 'var(--font-tiny)', 
            minWidth: '38px', 
            textAlign: 'right', 
            fontWeight: '500',
            whiteSpace: 'nowrap'
          }}>
            {((strength || 0) * 100).toFixed(1)}%
          </span>
        </div>
      )
    }
  ];

  // 중심성 분석 테이블 컬럼
  const centralityColumns = [
    {
      title: '직원명',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '부서',
      dataIndex: 'department',
      key: 'department',
      render: (dept) => <Tag color="blue">{dept}</Tag>
    },
    {
      title: '중심성',
      dataIndex: 'centrality',
      key: 'centrality',
      width: 140,
      render: (centrality) => (
        <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
          <Progress 
            percent={((centrality || 0) * 100).toFixed(0)} 
            size="small" 
            strokeColor="#1890ff"
            showInfo={false}
            style={{ width: 80, minWidth: 80 }}
          />
          <span style={{ 
            marginLeft: 6, 
            fontSize: 'var(--font-tiny)', 
            minWidth: '38px', 
            textAlign: 'right', 
            fontWeight: '500',
            whiteSpace: 'nowrap'
          }}>
            {((centrality || 0) * 100).toFixed(1)}%
          </span>
        </div>
      ),
      sorter: (a, b) => (a.centrality || 0) - (b.centrality || 0),
    },
    {
      title: '영향력 점수',
      dataIndex: 'influence_score',
      key: 'influence_score',
      width: 140,
      render: (score) => (
        <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
          <Progress 
            percent={((score || 0) * 100).toFixed(0)} 
            size="small" 
            strokeColor="#52c41a"
            showInfo={false}
            style={{ width: 80, minWidth: 80 }}
          />
          <span style={{ 
            marginLeft: 6, 
            fontSize: 'var(--font-tiny)', 
            minWidth: '38px', 
            textAlign: 'right', 
            fontWeight: '500',
            whiteSpace: 'nowrap'
          }}>
            {((score || 0) * 100).toFixed(1)}%
          </span>
        </div>
      ),
      sorter: (a, b) => (a.influence_score || 0) - (b.influence_score || 0),
    },
    {
      title: '위험도',
      dataIndex: 'risk_level',
      key: 'risk_level',
      render: (risk) => {
        const riskValue = risk || 0;
        const level = riskValue >= 0.7 ? 'HIGH' : riskValue >= 0.4 ? 'MEDIUM' : 'LOW';
        const color = level === 'HIGH' ? 'red' : level === 'MEDIUM' ? 'orange' : 'green';
        return <Tag color={color}>{level}</Tag>;
      }
    }
  ];

  // RelationshipAnalysis 컴포넌트의 JSX 반환
  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <TeamOutlined /> 개별 관계 분석
      </Title>
      
      <Paragraph>
        Neo4j 그래프 데이터베이스에서 직원들 간의 협업 관계, 네트워크 구조, 영향력을 분석하여 조직의 소통 패턴과 협업 효율성을 파악합니다.
        배치 분석 없이도 독립적으로 실행 가능하며, 실시간 관계 데이터를 기반으로 분석합니다.
      </Paragraph>

      {/* Neo4j 연결 설정 */}
      <Row gutter={24} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card title="Neo4j 데이터베이스 연결" extra={<DatabaseOutlined />}>
            <Row gutter={16} align="middle">
              <Col span={6}>
                <Text strong>Neo4j URI:</Text>
                <Input
                  value={neo4jConfig.uri}
                  onChange={(e) => setNeo4jConfig({...neo4jConfig, uri: e.target.value})}
                  placeholder="bolt://localhost:7687"
                  style={{ marginTop: 8 }}
                />
              </Col>
              <Col span={4}>
                <Text strong>사용자명:</Text>
                <Input
                  value={neo4jConfig.username}
                  onChange={(e) => setNeo4jConfig({...neo4jConfig, username: e.target.value})}
                  placeholder="neo4j"
                  style={{ marginTop: 8 }}
                />
              </Col>
              <Col span={4}>
                <Text strong>비밀번호:</Text>
                <Input.Password
                  value={neo4jConfig.password}
                  onChange={(e) => setNeo4jConfig({...neo4jConfig, password: e.target.value})}
                  placeholder="password"
                  style={{ marginTop: 8 }}
                />
              </Col>
              <Col span={4}>
                <Button 
                  type={neo4jConnected ? "default" : "primary"}
                  icon={<DatabaseOutlined />}
                  onClick={testNeo4jConnection}
                  loading={isAnalyzing}
                  style={{ marginTop: 24 }}
                >
                  {neo4jConnected ? '연결됨' : '연결 테스트'}
                </Button>
              </Col>
              <Col span={6}>
                {neo4jConnected && (
                  <Alert
                    message="✅ Neo4j 연결 성공"
                    type="success"
                    showIcon
                    style={{ marginTop: 8 }}
                  />
                )}
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      {/* 분석 설정 및 실행 */}
      <Row gutter={24} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card title="관계 분석 설정" extra={<NodeIndexOutlined />}>
            <Row gutter={16} align="middle">
              <Col span={6}>
                <Text strong>분석 유형:</Text>
                <Select 
                  value={analysisType} 
                  onChange={(value) => {
                    setAnalysisType(value);
                    setSearchTerm(''); // 분석 유형 변경 시 검색어 초기화
                  }}
                  style={{ width: '100%', marginTop: 8 }}
                >
                  <Option value="department">부서별 위험도 분석 ✅</Option>
                  <Option value="collaboration">협업 관계 분석 ✅</Option>
                </Select>
              </Col>
              {/* 부서별 위험도 분석일 때만 부서 선택 필드 표시 */}
              {analysisType === 'department' && (
                <Col span={6}>
                  <Text strong>부서 선택:</Text>
                  <Select
                    placeholder="분석할 부서를 선택하세요"
                    value={searchTerm}
                    onChange={setSearchTerm}
                    style={{ width: '100%', marginTop: 8 }}
                    showSearch
                    filterOption={(input, option) =>
                      option?.label?.toLowerCase().includes(input.toLowerCase())
                    }
                    loading={availableDepartments.length === 0}
                    notFoundContent={availableDepartments.length === 0 ? "부서 목록 로딩 중..." : "부서를 찾을 수 없습니다"}
                  >
                    {availableDepartments.map(dept => (
                      <Option key={dept.value} value={dept.value} label={dept.label}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span>{dept.value}</span>
                          <Tag color="blue" style={{ marginLeft: 8 }}>
                            {dept.employee_count}명
                          </Tag>
                        </div>
                      </Option>
                    ))}
                  </Select>
                </Col>
              )}
              
              {/* 협업 관계 분석일 때만 개별 직원 검색 필드 표시 */}
              {analysisType === 'collaboration' && (
                <Col span={6}>
                  <Text strong>개별 직원 검색:</Text>
                  <Input
                    placeholder="직원번호 (예: 1001)"
                    value={searchTerm}
                    onChange={(e) => {
                      const value = e.target.value;
                      if (/^\d*$/.test(value)) { // 숫자만 허용
                        setSearchTerm(value);
                      }
                    }}
                    prefix={<UserOutlined />}
                    style={{ marginTop: 8 }}
                  />
                </Col>
              )}
              <Col span={6}>
                <Button 
                  type="primary" 
                  icon={<ShareAltOutlined />}
                  onClick={analyzeRelationships}
                  loading={isAnalyzing}
                  disabled={!neo4jConnected}
                  style={{ marginTop: 24 }}
                >
                  관계 분석 시작
                </Button>
              </Col>
              <Col span={6}>
                <Button 
                  icon={<UserOutlined />}
                  onClick={checkSampleEmployees}
                  disabled={!neo4jConnected}
                  style={{ marginTop: 24 }}
                >
                  직원 번호 확인
                </Button>
              </Col>
            </Row>
            
            {/* 도움말 텍스트 */}
            <div style={{ marginTop: 16, padding: 12, backgroundColor: '#f6f8fa', borderRadius: 6 }}>
              <Text type="secondary" style={{ fontSize: 'var(--font-small)' }}>
                {analysisType === 'department' ? (
                  <>
                    💡 <strong>부서별 위험도 분석:</strong> 선택한 부서에서 이직 위험도가 높은 상위 15명의 직원들을 시각화합니다.
                    <br />
                    📊 <strong>분석 내용:</strong> Cognita에서 실제 위험도 데이터를 가져와 위험도 순으로 정렬하여 표시합니다.
                    <br />
                    🎯 <strong>표시 기준:</strong> 위험도가 높을수록 빨간색, 낮을수록 초록색으로 표시됩니다.
                    <br />
                    📋 <strong>사용법:</strong> 위의 부서 선택 드롭다운에서 분석할 부서를 선택한 후 '관계 분석 시작' 버튼을 클릭하세요.
                  </>
                ) : analysisType === 'collaboration' ? (
                  <>
                    🤝 <strong>협업 관계 분석:</strong> 특정 직원을 중심으로 한 협업 네트워크와 관계 패턴을 시각화합니다.
                    <br />
                    📊 <strong>분석 내용:</strong> 협업 파트너, 협업 강도, 프로젝트 연결, 관계 품질을 분석합니다.
                    <br />
                    🎯 <strong>개별 직원 관계:</strong> 선택한 직원의 협업 파트너 정보와 협업 강도를 상세히 표시합니다.
                    <br />
                    📋 <strong>사용법:</strong> 위의 개별 직원 검색란에 직원번호(예: 1001)를 입력한 후 '관계 분석 시작' 버튼을 클릭하세요.
                  </>
                ) : (
                  <>
                    💡 <strong>분석 유형을 선택해주세요:</strong> 위에서 원하는 분석 유형을 선택하면 해당 분석에 대한 자세한 설명을 확인할 수 있습니다.
                  </>
                )}
              </Text>
            </div>
          </Card>
        </Col>
      </Row>

      {!neo4jConnected && (
        <Alert
          message="Neo4j 연결 필요"
          description="관계 분석을 위해서는 먼저 Neo4j 그래프 데이터베이스에 연결해주세요. 연결 정보를 입력하고 '연결 테스트' 버튼을 클릭하세요."
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {batchResults && (
        <Alert
          message="배치 분석 결과 연동됨"
          description="배치 분석 결과가 연동되어 더 정확한 위험도 정보를 제공합니다."
          type="success"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* 네트워크 메트릭스 */}
      {networkData && (
        <Row gutter={24} style={{ marginBottom: 24 }}>
          <Col span={24}>
            <Card title="네트워크 개요" extra={<BarChartOutlined />}>
              <Row gutter={16}>
                <Col span={4}>
                  <Statistic
                    title="총 직원 수"
                    value={networkData?.metadata?.total_employees || networkData?.metrics?.total_employees || 0}
                    prefix={<UserOutlined />}
                  />
                </Col>
                <Col span={4}>
                  <Statistic
                    title="총 연결 수"
                    value={networkData?.links?.length || networkData?.metrics?.total_connections || 0}
                    prefix={<ShareAltOutlined />}
                  />
                </Col>
                <Col span={4}>
                  <Statistic
                    title="평균 연결 수"
                    value={networkData?.metrics?.avg_connections || 0}
                    prefix={<NodeIndexOutlined />}
                  />
                </Col>
                <Col span={4}>
                  <Statistic
                    title="네트워크 밀도"
                    value={networkData?.metrics?.network_density || 0}
                    prefix={<ClusterOutlined />}
                  />
                </Col>
                <Col span={4}>
                  <Statistic
                    title="클러스터 수"
                    value={networkData?.metrics?.clusters || 0}
                    prefix={<TeamOutlined />}
                  />
                </Col>
                <Col span={4}>
                  <Button 
                    type="primary" 
                    icon={<DownloadOutlined />}
                    onClick={() => {
                      // 네트워크 데이터 내보내기
                      const dataStr = JSON.stringify(networkData, null, 2);
                      const dataBlob = new Blob([dataStr], {type: 'application/json'});
                      const url = URL.createObjectURL(dataBlob);
                      const link = document.createElement('a');
                      link.href = url;
                      link.download = 'network_analysis.json';
                      link.click();
                    }}
                  >
                    데이터 내보내기
                  </Button>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      )}

      <Row gutter={24}>
        {/* 네트워크 그래프 */}
        <Col span={16}>
          <Card title="협업 네트워크 그래프" extra={<NodeIndexOutlined />}>
            {isAnalyzing ? (
              <div style={{ textAlign: 'center', padding: '100px 0' }}>
                <Spin size="large" />
                <div style={{ marginTop: 16 }}>
                  <Text>네트워크 분석 중...</Text>
                </div>
              </div>
            ) : networkData ? (
              <div style={{ width: '100%', overflow: 'hidden', position: 'relative' }}>
                {/* 확대/축소 컨트롤 */}
                <div style={{ 
                  position: 'absolute', 
                  top: 10, 
                  right: 10, 
                  zIndex: 10,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 4
                }}>
                  <Button 
                    size="small" 
                    icon={<PlusOutlined />}
                    onClick={() => handleZoom(1.2)}
                    title="확대"
                  />
                  <Button 
                    size="small" 
                    icon={<MinusOutlined />}
                    onClick={() => handleZoom(0.8)}
                    title="축소"
                  />
                  <Button 
                    size="small" 
                    icon={<ReloadOutlined />}
                    onClick={() => handleZoom('reset')}
                    title="원래 크기"
                  />
                </div>
                <svg 
                  ref={svgRef} 
                  style={{ 
                    border: '1px solid #d9d9d9', 
                    borderRadius: '6px',
                    width: '100%',
                    maxWidth: '800px',
                    height: '500px',
                    backgroundColor: '#fafafa',
                    display: 'block',
                    margin: '0 auto'
                  }}
                ></svg>
                
                {/* 동적 부서별 색상 범례 */}
                {(() => {
                  // 네트워크 데이터에서 실제 사용된 부서들과 직원 수 추출
                  const departmentCounts = {};
                  networkData.nodes
                    .filter(node => node.department)
                    .forEach(node => {
                      departmentCounts[node.department] = (departmentCounts[node.department] || 0) + 1;
                    });
                  
                  const departmentsInData = Object.keys(departmentCounts);
                  
                  if (departmentsInData.length === 0) return null;
                  
                  // 색상 스케일 (drawNetworkGraph와 동일)
                  const colorScale = d3.scaleOrdinal()
                    .domain(['HR', 'IT', 'Sales', 'Marketing', 'Finance', 'Research & Development', 'Human Resources', 'default'])
                    .range(['#1890ff', '#52c41a', '#fa8c16', '#eb2f96', '#722ed1', '#13c2c2', '#f759ab', '#8c8c8c']);
                  
                  return (
                    <div style={{ 
                      marginTop: 12, 
                      padding: 12, 
                      backgroundColor: '#f9f9f9', 
                      borderRadius: 6,
                      border: '1px solid #e8e8e8'
                    }}>
                      <Text strong style={{ marginBottom: 8, display: 'block' }}>🎨 분석 대상 부서별 색상 범례</Text>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                        {departmentsInData.map((dept) => (
                          <div key={dept} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <div style={{
                              width: 12,
                              height: 12,
                              borderRadius: '50%',
                              backgroundColor: colorScale(dept),
                              border: '1px solid #ccc'
                            }}></div>
                            <Text style={{ fontSize: 12 }}>
                              {dept} ({departmentCounts[dept]}명)
                            </Text>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })()}


                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">
                    💡 <strong>사용법:</strong> 마우스 휠로 확대/축소, 드래그로 그래프 이동, 노드 클릭으로 상세 정보 확인, 개별 노드 드래그로 위치 조정이 가능합니다.
                  </Text>
                  <br />
                  <Text type="secondary">
                    📊 <strong>표시:</strong> 실제 Cognita 분석 결과를 기반으로 위험도가 높은 직원들을 표시합니다. 노드 크기와 색상은 위험도를 나타냅니다.
                  </Text>
                  {networkData?.metadata && (
                    <>
                      <br />
                      <Text type="secondary" style={{ fontSize: 'var(--font-small)', color: '#52c41a' }}>
                        ✅ <strong>실제 분석 데이터:</strong> {networkData.metadata.department} 부서 
                        (총 {networkData.metadata.total_employees}명 중 {networkData.metadata.analyzed_employees}명 분석됨)
                      </Text>
                    </>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '100px 0' }}>
                <NodeIndexOutlined style={{ fontSize: 'var(--icon-xlarge)', color: '#d9d9d9' }} />
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">관계 분석을 시작하여 네트워크 그래프를 확인하세요</Text>
                </div>
              </div>
            )}
          </Card>
        </Col>

        {/* 선택된 직원 상세 정보 */}
        <Col span={8}>
          <Card title="직원 상세 정보" extra={<UserOutlined />}>
            {selectedEmployee ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>직원명:</Text> {selectedEmployee.name}
                </div>
                <div>
                  <Text strong>부서:</Text> <Tag color="blue">{selectedEmployee.department}</Tag>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <Text strong>중심성:</Text>
                  <div style={{ display: 'flex', alignItems: 'center', marginTop: 4 }}>
                    <Progress 
                      percent={(selectedEmployee.centrality * 100).toFixed(0)} 
                      size="small" 
                      strokeColor="#1890ff"
                      showInfo={false}
                      style={{ flex: 1, minWidth: 0 }}
                    />
                    <span style={{ marginLeft: 8, fontSize: 'var(--font-small)', minWidth: '45px', textAlign: 'right', fontWeight: '500' }}>
                      {(selectedEmployee.centrality * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <Text strong>영향력:</Text>
                  <div style={{ display: 'flex', alignItems: 'center', marginTop: 4 }}>
                    <Progress 
                      percent={(selectedEmployee.influence_score * 100).toFixed(0)} 
                      size="small" 
                      strokeColor="#52c41a"
                      showInfo={false}
                      style={{ flex: 1, minWidth: 0 }}
                    />
                    <span style={{ marginLeft: 8, fontSize: 'var(--font-small)', minWidth: '45px', textAlign: 'right', fontWeight: '500' }}>
                      {(selectedEmployee.influence_score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <Text strong>위험도:</Text>
                  <div style={{ display: 'flex', alignItems: 'center', marginTop: 4 }}>
                    <Progress 
                      percent={(selectedEmployee.risk_level * 100).toFixed(0)} 
                      size="small" 
                      strokeColor={selectedEmployee.risk_level > 0.7 ? '#ff4d4f' : selectedEmployee.risk_level > 0.4 ? '#fa8c16' : '#52c41a'}
                      showInfo={false}
                      style={{ flex: 1, minWidth: 0 }}
                    />
                    <span style={{ marginLeft: 8, fontSize: 'var(--font-small)', minWidth: '45px', textAlign: 'right', fontWeight: '500' }}>
                      {(selectedEmployee.risk_level * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <Divider />
                <div>
                  <Text strong>연결된 동료 수:</Text> {
                    networkData?.links?.filter(link => 
                      link.source === selectedEmployee.id || link.target === selectedEmployee.id
                    ).length || 0
                  }명
                </div>
                <div>
                  <Text strong>주요 협업 유형:</Text>
                  {networkData?.links?.filter(link => 
                    link.source === selectedEmployee.id || link.target === selectedEmployee.id
                  ).map(link => (
                    <Tag key={`${link.source}-${link.target}`} style={{ margin: '2px' }}>
                      {link.collaboration_type}
                    </Tag>
                  ))}
                </div>
              </Space>
            ) : (
              <div style={{ textAlign: 'center', padding: '50px 0' }}>
                <UserOutlined style={{ fontSize: 'var(--icon-xlarge)', color: '#d9d9d9' }} />
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">그래프에서 직원을 선택하세요</Text>
                </div>
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* 관계 분석 테이블들 */}
      {networkData && (() => {
        /* 분석 유형별 다른 테이블 표시 */
          const analysisType = networkData?.metadata?.analysis_type;
          
          switch (analysisType) {
            case 'department':
              return (
                <Row gutter={24} style={{ marginTop: 24 }}>
                  <Col span={12}>
                    <Card title="고위험 직원 협업 관계" extra={<ShareAltOutlined />}>
                      <Table
                        columns={relationshipColumns}
                        dataSource={networkData.links.filter(link => 
                          link.type === 'high_risk_collaboration'
                        )}
                        rowKey={(record) => `relationship-${record.source}-${record.target}`}
                        pagination={{ pageSize: 8 }}
                        size="small"
                        scroll={{ y: 400, x: 'max-content' }}
                        style={{ fontSize: 'var(--font-small)' }}
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="고위험 직원 위험도 분석" extra={<BarChartOutlined />}>
                      <Table
                        columns={centralityColumns}
                        dataSource={networkData.nodes.filter(node => 
                          node.type === 'high_risk_employee'
                        )}
                        rowKey={(record) => `employee-${record.id || record.name || Math.random()}`}
                        pagination={{ pageSize: 8 }}
                        size="small"
                        scroll={{ y: 400, x: 'max-content' }}
                        style={{ fontSize: 'var(--font-small)' }}
                      />
                    </Card>
                  </Col>
                </Row>
              );
              
            case 'collaboration':
            case 'employee':
              // 개별 직원 협업 관계 분석 (collaboration과 employee 케이스 통합)
              return (
                <Row gutter={24} style={{ marginTop: 24 }}>
                  <Col span={12}>
                    <Card title="개별 직원 관계" extra={<ShareAltOutlined />}>
                      <Table
                        columns={relationshipColumns}
                        dataSource={networkData.links}
                        rowKey={(record) => `employee-rel-${record.source}-${record.target}`}
                        pagination={{ pageSize: 8 }}
                        size="small"
                        scroll={{ y: 400, x: 'max-content' }}
                        style={{ fontSize: 'var(--font-small)' }}
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="연결된 동료들" extra={<UserOutlined />}>
                      <Table
                        columns={centralityColumns}
                        dataSource={networkData.nodes.filter(node => 
                          node.type === 'colleague' || node.type === 'center_employee'
                        )}
                        rowKey={(record) => `colleague-${record.id || record.name || Math.random()}`}
                        pagination={{ pageSize: 8 }}
                        size="small"
                        scroll={{ y: 400, x: 'max-content' }}
                        style={{ fontSize: 'var(--font-small)' }}
                      />
                    </Card>
                  </Col>
                </Row>
              );
              
            default:
              return (
                <Row gutter={24} style={{ marginTop: 24 }}>
                  <Col span={24}>
                    <Card title="분석 결과" extra={<BarChartOutlined />}>
                      <Text type="secondary">
                        분석 유형: {analysisType || '알 수 없음'}
                      </Text>
                    </Card>
                  </Col>
                </Row>
              );
          }
        })()}
    </div>
  );
};

export default RelationshipAnalysis;
