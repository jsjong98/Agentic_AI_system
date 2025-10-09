import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Space,
  Select,
  Typography,
  Progress,
  Spin,
  message
} from 'antd';
import {
  TeamOutlined,
  BarChartOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  ReloadOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Option } = Select;

const GroupStatistics = ({ 
  loading, 
  setLoading, 
  serverStatus, 
  globalBatchResults, 
  lastAnalysisTimestamp 
}) => {
  const [statistics, setStatistics] = useState(null);
  const [groupBy, setGroupBy] = useState('department');
  const [departmentFilter, setDepartmentFilter] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [availableDepartments, setAvailableDepartments] = useState([]);
  const [dataSource, setDataSource] = useState('server'); // 'server' 또는 'batch'

  // 컴포넌트 마운트 시 데이터 로드
  useEffect(() => {
    // 배치 분석 결과가 있으면 우선 사용
    if (globalBatchResults && globalBatchResults.results) {
      console.log('배치 분석 결과를 사용하여 통계 생성:', globalBatchResults);
      generateStatisticsFromBatchResults();
      setDataSource('batch');
    } else {
      // 배치 결과가 없으면 서버에서 로드
      loadStatistics();
      setDataSource('server');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [globalBatchResults]);

  // 배치 분석 결과에서 직원 정보 가져오기
  const getEmployeeMetadata = (employeeNumber, employeeData) => {
    try {
      // 1. 현재 직원 데이터에서 직접 추출
      if (employeeData) {
        // 직원 데이터에서 메타데이터 추출 (여러 경로 시도)
        const department = employeeData.department || 
                          employeeData.Department ||
                          employeeData.analysis_result?.employee_data?.Department ||
                          employeeData.employee_data?.Department ||
                          'Unknown';
        
        const job_role = employeeData.job_role || 
                        employeeData.JobRole ||
                        employeeData.analysis_result?.employee_data?.JobRole ||
                        employeeData.employee_data?.JobRole ||
                        'Unknown';
        
        const position = employeeData.position || 
                        employeeData.Position ||
                        employeeData.job_level ||
                        employeeData.JobLevel ||
                        employeeData.analysis_result?.employee_data?.JobLevel ||
                        employeeData.employee_data?.JobLevel ||
                        'Unknown';
        
        return {
          department,
          job_role,
          position
        };
      }
      
      // 2. globalBatchResults에서 찾기 (fallback)
      const results = Array.isArray(globalBatchResults) ? globalBatchResults : 
                     (globalBatchResults && globalBatchResults.results) ? globalBatchResults.results : [];
      
      if (results && results.length > 0) {
        const employee = results.find(emp => 
          emp.employee_number === employeeNumber || 
          emp.employee_id === employeeNumber ||
          emp.id === employeeNumber ||
          String(emp.employee_number) === String(employeeNumber)
        );
        
        if (employee) {
          const department = employee.department || 
                            employee.Department ||
                            employee.analysis_result?.employee_data?.Department ||
                            'Unknown';
          
          const job_role = employee.job_role || 
                          employee.JobRole ||
                          employee.analysis_result?.employee_data?.JobRole ||
                          'Unknown';
          
          const position = employee.position || 
                          employee.Position ||
                          employee.job_level ||
                          employee.JobLevel ||
                          employee.analysis_result?.employee_data?.JobLevel ||
                          'Unknown';
          
          return {
            department,
            job_role,
            position
          };
        }
      }
      
      // 3. 찾지 못한 경우 기본값 반환 (경고 없이)
      return {
        department: 'Unknown',
        job_role: 'Unknown',
        position: 'Unknown'
      };
    } catch (error) {
      console.error('직원 메타데이터 조회 실패:', error);
      return {
        department: 'Unknown',
        job_role: 'Unknown',
        position: 'Unknown'
      };
    }
  };

  // 배치 분석 결과로부터 통계 생성
  const generateStatisticsFromBatchResults = () => {
    // globalBatchResults가 배열인지 객체인지 확인
    const results = Array.isArray(globalBatchResults) ? globalBatchResults : 
                   (globalBatchResults && globalBatchResults.results) ? globalBatchResults.results : [];
    
    if (!results || results.length === 0) {
      console.warn('배치 분석 결과가 없습니다.');
      return;
    }
    
    console.log('📊 배치 결과 통계 생성 시작:', {
      totalResults: results.length,
      firstEmployee: results[0],
      groupBy: groupBy
    });
    
    setIsLoading(true);
    try {
      const groupedStats = {};
      
      // 그룹화 로직 개선 (직접 메타데이터 추출)
      results.forEach(employee => {
        let groupKey = 'Unknown';
        const employeeNumber = employee.employee_number;
        
        // 현재 직원 데이터에서 직접 메타데이터 추출
        const metadata = getEmployeeMetadata(employeeNumber, employee);
        
        // 부서별 그룹화
        if (groupBy === 'department') {
          groupKey = metadata.department;
        }
        // 직무별 그룹화
        else if (groupBy === 'job_role') {
          groupKey = metadata.job_role;
        }
        // 직급별 그룹화 (position 사용)
        else if (groupBy === 'job_level') {
          groupKey = metadata.position;
          
          // JobLevel이 숫자인 경우 텍스트로 변환
          if (typeof groupKey === 'number') {
            const levelMap = {
              1: 'Level 1 (Junior)',
              2: 'Level 2 (Associate)',
              3: 'Level 3 (Senior)',
              4: 'Level 4 (Lead)',
              5: 'Level 5 (Manager)'
            };
            groupKey = levelMap[groupKey] || `Level ${groupKey}`;
          }
        }
        
        console.log(`👤 직원 ${employeeNumber}: ${groupBy}=${groupKey}, 메타데이터:`, metadata);
        
        if (!groupedStats[groupKey]) {
          groupedStats[groupKey] = {
            total_employees: 0,
            high_risk: 0,
            medium_risk: 0,
            low_risk: 0,
            avg_risk_score: 0,
            common_risk_factors: {},
            risk_scores: []
          };
        }
          
        groupedStats[groupKey].total_employees++;
        
        // 위험도 점수 추출 - 여러 경로 시도
        let riskScore = 0;
        
        // 1. 저장된 risk_score 사용 (배치 분석 결과)
        if (employee.risk_score && employee.risk_score > 0) {
          riskScore = employee.risk_score;
        }
        // 2. combined_analysis 경로 시도
        else if (employee.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score) {
          riskScore = employee.analysis_result.combined_analysis.integrated_assessment.overall_risk_score;
        }
        // 3. 개별 에이전트 점수들로 직접 계산
        else if (employee.agent_results || employee.analysis_result) {
          const agentResults = employee.agent_results || employee.analysis_result;
          
          // 각 에이전트 점수 추출
          const structuraScore = agentResults.structura?.attrition_probability || 0;
          const chronosScore = agentResults.chronos?.risk_score || 0;
          const cognitaScore = agentResults.cognita?.overall_risk_score || 0;
          const sentioScore = agentResults.sentio?.risk_score || 0;
          const agoraScore = agentResults.agora?.market_risk_score || 0;
          
          // 간단한 평균으로 통합 (실제로는 가중평균을 사용해야 함)
          const scores = [structuraScore, chronosScore, cognitaScore, sentioScore, agoraScore];
          const validScores = scores.filter(score => score > 0);
          
          if (validScores.length > 0) {
            riskScore = validScores.reduce((sum, score) => sum + score, 0) / validScores.length;
          }
        }
        
        groupedStats[groupKey].risk_scores.push(riskScore);
        
        console.log(`🔍 직원 ${employee.employee_number} 위험도:`, riskScore, `${groupBy}:`, groupKey);
        
        // 위험도 분류 (0-1 범위 기준) - 배치 분석과 동일한 임계값 사용
        if (riskScore >= 0.7) {
          groupedStats[groupKey].high_risk++;
        } else if (riskScore >= 0.3) {
          groupedStats[groupKey].medium_risk++;
        } else {
          groupedStats[groupKey].low_risk++;
        }
      });
      
      // 평균 위험도 계산 및 최종 통계 정리
      Object.keys(groupedStats).forEach(groupKey => {
        const scores = groupedStats[groupKey].risk_scores;
        if (scores.length > 0) {
          groupedStats[groupKey].avg_risk_score = scores.reduce((sum, score) => sum + score, 0) / scores.length;
        } else {
          groupedStats[groupKey].avg_risk_score = 0;
        }
        
        console.log(`📊 ${groupKey} ${groupBy} 최종 통계:`, {
          total: groupedStats[groupKey].total_employees,
          high: groupedStats[groupKey].high_risk,
          medium: groupedStats[groupKey].medium_risk,
          low: groupedStats[groupKey].low_risk,
          avgScore: groupedStats[groupKey].avg_risk_score
        });
        
        delete groupedStats[groupKey].risk_scores; // 임시 배열 제거
      });
      
      // 사용 가능한 그룹 목록 업데이트
      if (groupBy === 'department') {
        setAvailableDepartments(Object.keys(groupedStats));
      }
      
      setStatistics({
        group_by: groupBy,
        department_filter: departmentFilter,
        statistics: groupedStats,
        generated_at: new Date().toISOString(),
        data_source: 'batch_analysis'
      });
      
    } catch (error) {
      console.error('배치 결과 통계 생성 실패:', error);
      message.error('통계 생성 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  const loadStatistics = async (newGroupBy = 'department', newDepartment = null) => {
    setIsLoading(true);
    try {
      const params = new URLSearchParams({
        group_by: newGroupBy
      });
      
      if (newDepartment) {
        params.append('department', newDepartment);
      }

      // 먼저 저장된 파일에서 로드 시도 (Integration 서버 5007)
      console.log('📁 저장된 파일에서 통계 로드 시도...');
      let response = await fetch(`http://localhost:5007/api/statistics/load-from-files?${params}`);
      
      if (!response.ok) {
        console.log('📁 저장된 파일 로드 실패, 기존 API 시도...');
        // 저장된 파일 로드 실패 시 기존 API 사용 (Supervisor 5006)
        response = await fetch(`http://localhost:5006/api/statistics/group?${params}`);
        
        if (!response.ok) {
          throw new Error('통계 로드 실패');
        }
      }

      const data = await response.json();
      console.log('📊 통계 로드 성공:', data);
      setStatistics(data);
      
      // 부서 목록 업데이트 (부서별 통계일 때)
      if (newGroupBy === 'department') {
        setAvailableDepartments(Object.keys(data.statistics));
      }
      
    } catch (error) {
      console.error('통계 로드 실패:', error);
      message.error('서버에서 통계 데이터를 가져오는데 실패했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGroupByChange = (value) => {
    setGroupBy(value);
    setDepartmentFilter(null);
    
    // 데이터 소스에 따라 다른 함수 호출
    if (dataSource === 'batch') {
      generateStatisticsFromBatchResults();
    } else {
      loadStatistics(value, null);
    }
  };

  const handleDepartmentFilterChange = (value) => {
    setDepartmentFilter(value);
    
    if (dataSource === 'batch') {
      generateStatisticsFromBatchResults();
    } else {
      loadStatistics(groupBy, value);
    }
  };

  const handleRefresh = () => {
    if (dataSource === 'batch' && globalBatchResults) {
      generateStatisticsFromBatchResults();
    } else {
      loadStatistics(groupBy, departmentFilter);
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'HIGH': return '#ff4d4f';
      case 'MEDIUM': return '#faad14';
      case 'LOW': return '#52c41a';
      default: return '#d9d9d9';
    }
  };

  const getRiskLevelFromScore = (score) => {
    if (score >= 0.7) return 'HIGH';
    if (score >= 0.3) return 'MEDIUM';
    return 'LOW';
  };

  // 테이블 컬럼 정의
  const getTableColumns = () => {
    const baseColumns = [
      {
        title: groupBy === 'department' ? '부서명' : 
               groupBy === 'job_role' ? '직무명' : '직급명',
        dataIndex: 'name',
        key: 'name',
        render: (text) => <Text strong>{text}</Text>
      },
      {
        title: '총 직원 수',
        dataIndex: 'total_employees',
        key: 'total_employees',
        render: (count) => <Statistic value={count} suffix="명" />
      },
      {
        title: '평균 위험도',
        dataIndex: 'avg_risk_score',
        key: 'avg_risk_score',
        render: (score) => (
          <Space>
            <Progress 
              percent={Math.round(score * 100)} 
              size="small" 
              strokeColor={getRiskColor(getRiskLevelFromScore(score))}
              showInfo={false}
              style={{ width: 80 }}
            />
            <Tag color={getRiskColor(getRiskLevelFromScore(score))}>
              {(score * 100).toFixed(1)}%
            </Tag>
          </Space>
        ),
        sorter: (a, b) => a.avg_risk_score - b.avg_risk_score
      },
      {
        title: '위험도 분포',
        key: 'risk_distribution',
        render: (_, record) => (
          <Space>
            <Tag color="red">{record.high_risk}명</Tag>
            <Tag color="orange">{record.medium_risk}명</Tag>
            <Tag color="green">{record.low_risk}명</Tag>
          </Space>
        )
      }
    ];

    // 부서별 통계일 때 공통 위험 요인 컬럼 추가
    if (groupBy === 'department') {
      baseColumns.push({
        title: '주요 위험 요인',
        dataIndex: 'common_risk_factors',
        key: 'common_risk_factors',
        render: (factors) => (
          <Space wrap>
            {Object.entries(factors || {}).slice(0, 3).map(([factor, count]) => (
              <Tag key={factor} color="volcano">
                {factor} ({count})
              </Tag>
            ))}
          </Space>
        )
      });
    }

    return baseColumns;
  };

  // 테이블 데이터 변환
  const getTableData = () => {
    if (!statistics || !statistics.statistics) return [];
    
    return Object.entries(statistics.statistics).map(([name, stats]) => ({
      key: name,
      name: name.replace(/_/g, ' '),
      ...stats
    }));
  };

  // 전체 요약 통계
  const getSummaryStats = () => {
    if (!statistics || !statistics.statistics) return null;
    
    const allStats = Object.values(statistics.statistics);
    const totalEmployees = allStats.reduce((sum, stat) => sum + stat.total_employees, 0);
    const totalHighRisk = allStats.reduce((sum, stat) => sum + stat.high_risk, 0);
    const totalMediumRisk = allStats.reduce((sum, stat) => sum + stat.medium_risk, 0);
    const totalLowRisk = allStats.reduce((sum, stat) => sum + stat.low_risk, 0);
    
    const avgRiskScore = totalEmployees > 0 ? 
      allStats.reduce((sum, stat) => sum + (stat.avg_risk_score * stat.total_employees), 0) / totalEmployees : 0;

    return {
      totalEmployees,
      totalHighRisk,
      totalMediumRisk,
      totalLowRisk,
      avgRiskScore,
      highRiskPercentage: totalEmployees > 0 ? (totalHighRisk / totalEmployees) * 100 : 0
    };
  };

  const summaryStats = getSummaryStats();

  return (
    <div style={{ padding: '24px' }}>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Title level={2}>
                  <BarChartOutlined /> 단체 통계 분석
                </Title>
                <div>
                  <Text type="secondary">
                    부서별, 직무별, 직급별 위험도 통계를 확인하세요
                  </Text>
                  {dataSource === 'batch' && lastAnalysisTimestamp && (
                    <div style={{ marginTop: '8px' }}>
                      <Tag color="green" icon={<CheckCircleOutlined />}>
                        배치 분석 결과 기반 ({new Date(lastAnalysisTimestamp).toLocaleString()})
                      </Tag>
                    </div>
                  )}
                  {dataSource === 'server' && statistics && (
                    <div style={{ marginTop: '8px' }}>
                      <Tag color={statistics.data_source === 'saved_files' ? 'purple' : 'blue'}>
                        {statistics.data_source === 'saved_files' ? '저장된 파일 기반' : '서버 저장 데이터 기반'}
                      </Tag>
                    </div>
                  )}
                </div>
              </Col>
              <Col>
                <Space>
                  <Select
                    value={groupBy}
                    onChange={handleGroupByChange}
                    style={{ width: 120 }}
                  >
                    <Option value="department">부서별</Option>
                    <Option value="job_role">직무별</Option>
                    <Option value="job_level">직급별</Option>
                  </Select>
                  
                  {groupBy === 'job_role' && (
                    <Select
                      placeholder="부서 선택"
                      value={departmentFilter}
                      onChange={handleDepartmentFilterChange}
                      style={{ width: 150 }}
                      allowClear
                    >
                      {availableDepartments.map(dept => (
                        <Option key={dept} value={dept}>{dept}</Option>
                      ))}
                    </Select>
                  )}
                  
                  <Button 
                    icon={<ReloadOutlined />}
                    onClick={handleRefresh}
                    loading={isLoading}
                  >
                    새로고침
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* 전체 요약 통계 */}
        {summaryStats && (
          <Col span={24}>
            <Card title="전체 요약">
              <Row gutter={16}>
                <Col span={6}>
                  <Statistic
                    title="총 직원 수"
                    value={summaryStats.totalEmployees}
                    suffix="명"
                    prefix={<TeamOutlined />}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="평균 위험도"
                    value={summaryStats.avgRiskScore * 100}
                    precision={1}
                    suffix="%"
                    valueStyle={{ 
                      color: getRiskColor(getRiskLevelFromScore(summaryStats.avgRiskScore))
                    }}
                    prefix={<BarChartOutlined />}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="고위험군"
                    value={summaryStats.totalHighRisk}
                    suffix="명"
                    valueStyle={{ color: '#cf1322' }}
                    prefix={<ExclamationCircleOutlined />}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="고위험군 비율"
                    value={summaryStats.highRiskPercentage}
                    precision={1}
                    suffix="%"
                    valueStyle={{ 
                      color: summaryStats.highRiskPercentage > 20 ? '#cf1322' : '#52c41a'
                    }}
                    prefix={<WarningOutlined />}
                  />
                </Col>
              </Row>
            </Card>
          </Col>
        )}

        {/* 상세 통계 테이블 */}
        <Col span={24}>
          <Card title={`${groupBy === 'department' ? '부서' : groupBy === 'job_role' ? '직무' : '직급'}별 상세 통계`}>
            {isLoading ? (
              <div style={{ textAlign: 'center', padding: '50px' }}>
                <Spin size="large" />
                <p style={{ marginTop: '16px' }}>통계 데이터를 불러오는 중...</p>
              </div>
            ) : (
              <Table
                columns={getTableColumns()}
                dataSource={getTableData()}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total) => `총 ${total}개 그룹`
                }}
                scroll={{ x: 800 }}
              />
            )}
          </Card>
        </Col>

        {/* 위험 요인 분석 (부서별일 때만) */}
        {groupBy === 'department' && statistics && (
          <Col span={24}>
            <Card title="부서별 주요 위험 요인 분석">
              <Row gutter={[16, 16]}>
                {Object.entries(statistics.statistics).map(([deptName, stats]) => (
                  <Col span={8} key={deptName}>
                    <Card size="small" title={deptName}>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Text strong>위험 요인 TOP 3:</Text>
                        {Object.entries(stats.common_risk_factors || {}).slice(0, 3).map(([factor, count]) => (
                          <div key={factor} style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Text>{factor}</Text>
                            <Tag color="red">{count}건</Tag>
                          </div>
                        ))}
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            </Card>
          </Col>
        )}
      </Row>
    </div>
  );
};

export default GroupStatistics;
