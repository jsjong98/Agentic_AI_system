import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Table,
  message,
  Typography,
  Row,
  Col,
  Statistic,
  Tag,
  Modal,
  Spin,
  Alert,
  Space,
  Select,
  Input,
  Divider
} from 'antd';
import {
  FileTextOutlined,
  UserOutlined,
  BarChartOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  DownloadOutlined,
  EyeOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { Option } = Select;

const ReportGeneration = () => {
  const [batchResults, setBatchResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [reportModalVisible, setReportModalVisible] = useState(false);
  const [generatedReport, setGeneratedReport] = useState('');
  const [reportGenerating, setReportGenerating] = useState(false);
  const [riskFilter, setRiskFilter] = useState('all');
  const [departmentFilter, setDepartmentFilter] = useState('all');

  // 컴포넌트 로드 시 배치 분석 결과 로드
  useEffect(() => {
    loadBatchResults();
  }, []);

  // 배치 분석 결과 로드
  const loadBatchResults = async () => {
    try {
      setLoading(true);
      
      // localStorage에서 배치 분석 결과 로드
      const savedResults = localStorage.getItem('batchAnalysisResults');
      if (savedResults) {
        const results = JSON.parse(savedResults);
        setBatchResults(results);
        console.log('📊 배치 분석 결과 로드됨:', results);
      } else {
        message.info('배치 분석 결과가 없습니다. 먼저 배치 분석을 실행해주세요.');
      }
    } catch (error) {
      console.error('배치 분석 결과 로드 실패:', error);
      message.error('배치 분석 결과를 불러올 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  // 위험도별 직원 분류
  const getEmployeesByRisk = () => {
    if (!batchResults || !batchResults.results) return { high: [], medium: [], low: [] };

    const employees = batchResults.results.map(emp => {
      const riskScore = emp.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 0;
      const department = emp.analysis_result?.employee_data?.Department || emp.department || '미분류';
      
      // 사후 분석 최적화 설정 적용
      const finalSettings = JSON.parse(localStorage.getItem('finalRiskSettings') || '{}');
      const highThreshold = finalSettings.risk_thresholds?.high_risk_threshold || 0.7;
      const lowThreshold = finalSettings.risk_thresholds?.low_risk_threshold || 0.3;
      
      let riskLevel = 'low';
      if (riskScore >= highThreshold) riskLevel = 'high';
      else if (riskScore >= lowThreshold) riskLevel = 'medium';

      return {
        key: emp.employee_number,
        employee_id: emp.employee_number,
        name: emp.analysis_result?.employee_data?.Name || `직원 ${emp.employee_number}`,
        department: department,
        risk_score: riskScore,
        risk_level: riskLevel,
        structura_score: emp.analysis_result?.structura_result?.prediction?.attrition_probability || 0,
        chronos_score: emp.analysis_result?.chronos_result?.prediction?.risk_score || 0,
        cognita_score: emp.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score || 0,
        sentio_score: emp.analysis_result?.sentio_result?.sentiment_analysis?.risk_score || 0,
        agora_score: emp.analysis_result?.agora_result?.market_analysis?.risk_score || 0
      };
    });

    return {
      high: employees.filter(emp => emp.risk_level === 'high'),
      medium: employees.filter(emp => emp.risk_level === 'medium'),
      low: employees.filter(emp => emp.risk_level === 'low')
    };
  };

  // 필터링된 직원 목록
  const getFilteredEmployees = () => {
    const employeesByRisk = getEmployeesByRisk();
    let allEmployees = [...employeesByRisk.high, ...employeesByRisk.medium, ...employeesByRisk.low];

    // 위험도 필터
    if (riskFilter !== 'all') {
      allEmployees = employeesByRisk[riskFilter] || [];
    }

    // 부서 필터
    if (departmentFilter !== 'all') {
      allEmployees = allEmployees.filter(emp => emp.department === departmentFilter);
    }

    return allEmployees;
  };

  // 부서 목록 추출
  const getDepartments = () => {
    if (!batchResults || !batchResults.results) return [];
    const departments = [...new Set(batchResults.results.map(emp => 
      emp.analysis_result?.employee_data?.Department || emp.department || '미분류'
    ))];
    return departments;
  };

  // 개별 직원 보고서 생성
  const generateEmployeeReport = async (employee) => {
    try {
      setReportGenerating(true);
      setSelectedEmployee(employee);
      setReportModalVisible(true);

      console.log('📝 직원 보고서 생성 시작:', employee.employee_id);

      // Integration 서버에 보고서 생성 요청
      const response = await fetch('http://localhost:5007/api/generate-employee-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          employee_id: employee.employee_id,
          department: employee.department,
          risk_level: employee.risk_level,
          risk_score: employee.risk_score,
          agent_scores: {
            structura: employee.structura_score,
            chronos: employee.chronos_score,
            cognita: employee.cognita_score,
            sentio: employee.sentio_score,
            agora: employee.agora_score
          }
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setGeneratedReport(result.report);
        console.log('✅ 직원 보고서 생성 완료');
      } else {
        throw new Error(result.error || '보고서 생성 실패');
      }

    } catch (error) {
      console.error('보고서 생성 실패:', error);
      message.error(`보고서 생성에 실패했습니다: ${error.message}`);
      setGeneratedReport('보고서 생성에 실패했습니다. 다시 시도해주세요.');
    } finally {
      setReportGenerating(false);
    }
  };

  // 테이블 컬럼 정의
  const columns = [
    {
      title: '직원 ID',
      dataIndex: 'employee_id',
      key: 'employee_id',
      width: 100,
    },
    {
      title: '이름',
      dataIndex: 'name',
      key: 'name',
      width: 120,
    },
    {
      title: '부서',
      dataIndex: 'department',
      key: 'department',
      width: 150,
    },
    {
      title: '위험도',
      dataIndex: 'risk_level',
      key: 'risk_level',
      width: 100,
      render: (level) => {
        const config = {
          high: { color: 'red', text: '고위험군' },
          medium: { color: 'orange', text: '주의군' },
          low: { color: 'green', text: '안전군' }
        };
        return <Tag color={config[level]?.color}>{config[level]?.text}</Tag>;
      }
    },
    {
      title: '위험 점수',
      dataIndex: 'risk_score',
      key: 'risk_score',
      width: 100,
      render: (score) => (score * 100).toFixed(1) + '%',
      sorter: (a, b) => a.risk_score - b.risk_score,
    },
    {
      title: 'Structura',
      dataIndex: 'structura_score',
      key: 'structura_score',
      width: 100,
      render: (score) => (score * 100).toFixed(1) + '%',
    },
    {
      title: 'Chronos',
      dataIndex: 'chronos_score',
      key: 'chronos_score',
      width: 100,
      render: (score) => (score * 100).toFixed(1) + '%',
    },
    {
      title: '액션',
      key: 'action',
      width: 120,
      render: (_, record) => (
        <Space>
          <Button
            type="primary"
            icon={<FileTextOutlined />}
            size="small"
            onClick={() => generateEmployeeReport(record)}
          >
            보고서
          </Button>
        </Space>
      ),
    },
  ];

  const employeesByRisk = getEmployeesByRisk();
  const filteredEmployees = getFilteredEmployees();

  if (loading) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text>배치 분석 결과를 불러오는 중...</Text>
        </div>
      </div>
    );
  }

  if (!batchResults) {
    return (
      <div style={{ padding: '24px' }}>
        <Alert
          message="배치 분석 결과 없음"
          description="보고서를 생성하려면 먼저 배치 분석을 실행해주세요."
          type="info"
          showIcon
          action={
            <Button size="small" onClick={loadBatchResults}>
              다시 시도
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <FileTextOutlined /> 보고서 출력
      </Title>
      
      <Paragraph>
        배치 분석 결과를 기반으로 개별 직원의 상세 보고서를 생성합니다.
        각 직원의 위험도 분석 결과와 XAI 설명을 포함한 종합 보고서를 제공합니다.
      </Paragraph>

      {/* 통계 요약 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Statistic
            title="총 직원 수"
            value={batchResults.total_employees || 0}
            prefix={<UserOutlined />}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="고위험군"
            value={employeesByRisk.high.length}
            valueStyle={{ color: '#cf1322' }}
            prefix={<ExclamationCircleOutlined />}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="주의군"
            value={employeesByRisk.medium.length}
            valueStyle={{ color: '#fa8c16' }}
            prefix={<ExclamationCircleOutlined />}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="안전군"
            value={employeesByRisk.low.length}
            valueStyle={{ color: '#52c41a' }}
            prefix={<CheckCircleOutlined />}
          />
        </Col>
      </Row>

      {/* 필터 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col span={8}>
            <Text strong>위험도 필터:</Text>
            <Select
              style={{ width: '100%', marginTop: 8 }}
              value={riskFilter}
              onChange={setRiskFilter}
            >
              <Option value="all">전체</Option>
              <Option value="high">고위험군</Option>
              <Option value="medium">주의군</Option>
              <Option value="low">안전군</Option>
            </Select>
          </Col>
          <Col span={8}>
            <Text strong>부서 필터:</Text>
            <Select
              style={{ width: '100%', marginTop: 8 }}
              value={departmentFilter}
              onChange={setDepartmentFilter}
            >
              <Option value="all">전체 부서</Option>
              {getDepartments().map(dept => (
                <Option key={dept} value={dept}>{dept}</Option>
              ))}
            </Select>
          </Col>
          <Col span={8}>
            <Text strong>필터링된 결과:</Text>
            <div style={{ marginTop: 8 }}>
              <Text>{filteredEmployees.length}명</Text>
            </div>
          </Col>
        </Row>
      </Card>

      {/* 직원 목록 테이블 */}
      <Card title="직원 목록" extra={<BarChartOutlined />}>
        <Table
          columns={columns}
          dataSource={filteredEmployees}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} / 총 ${total}명`,
          }}
          scroll={{ x: 800 }}
        />
      </Card>

      {/* 보고서 모달 */}
      <Modal
        title={`직원 보고서 - ${selectedEmployee?.name || selectedEmployee?.employee_id}`}
        open={reportModalVisible}
        onCancel={() => setReportModalVisible(false)}
        width={800}
        footer={[
          <Button key="close" onClick={() => setReportModalVisible(false)}>
            닫기
          </Button>,
          <Button
            key="download"
            type="primary"
            icon={<DownloadOutlined />}
            onClick={() => {
              // 보고서 다운로드 기능 (추후 구현)
              message.info('다운로드 기능은 추후 구현 예정입니다.');
            }}
          >
            다운로드
          </Button>,
        ]}
      >
        {reportGenerating ? (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <Spin size="large" />
            <div style={{ marginTop: 16 }}>
              <Text>LLM으로 보고서를 생성하는 중...</Text>
            </div>
          </div>
        ) : (
          <div>
            {selectedEmployee && (
              <div style={{ marginBottom: 16, padding: 16, backgroundColor: '#f5f5f5', borderRadius: 6 }}>
                <Row gutter={16}>
                  <Col span={12}>
                    <Text strong>직원 ID:</Text> {selectedEmployee.employee_id}<br />
                    <Text strong>부서:</Text> {selectedEmployee.department}<br />
                    <Text strong>위험도:</Text> <Tag color={
                      selectedEmployee.risk_level === 'high' ? 'red' : 
                      selectedEmployee.risk_level === 'medium' ? 'orange' : 'green'
                    }>
                      {selectedEmployee.risk_level === 'high' ? '고위험군' : 
                       selectedEmployee.risk_level === 'medium' ? '주의군' : '안전군'}
                    </Tag>
                  </Col>
                  <Col span={12}>
                    <Text strong>위험 점수:</Text> {(selectedEmployee.risk_score * 100).toFixed(1)}%<br />
                    <Text strong>Structura:</Text> {(selectedEmployee.structura_score * 100).toFixed(1)}%<br />
                    <Text strong>Chronos:</Text> {(selectedEmployee.chronos_score * 100).toFixed(1)}%
                  </Col>
                </Row>
              </div>
            )}
            
            <Divider>생성된 보고서</Divider>
            
            <TextArea
              value={generatedReport}
              readOnly
              rows={15}
              style={{ fontSize: '14px', lineHeight: '1.6' }}
              placeholder="보고서가 생성되면 여기에 표시됩니다..."
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default ReportGeneration;
