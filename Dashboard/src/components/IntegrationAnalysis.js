import React, { useState } from 'react';
import {
  Card,
  Button,
  Steps,
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
  Space
} from 'antd';
import {
  UploadOutlined,
  FileTextOutlined,
  CheckCircleOutlined,
  ApiOutlined,
  BarChartOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';
import { apiService } from '../services/apiService';

const { Title, Text, Paragraph } = Typography;
const { Dragger } = Upload;

const IntegrationAnalysis = ({
  loading,
  setLoading,
  serverStatus,
  stepStatuses,
  onIntegrationCompleted
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [uploadedFiles, setUploadedFiles] = useState({});
  const [analysisResults, setAnalysisResults] = useState(null);
  const [reportData, setReportData] = useState(null);
  const [progress, setProgress] = useState(0);

  // 필요한 파일들 정의 (올바른 순서: Structura → Cognita → Chronos → Sentio → Agora)
  const requiredFiles = [
    {
      key: 'structura_results',
      name: 'Structura 분석 결과',
      description: '정형 데이터 분석 결과 파일 (HR 이직 예측)',
      accept: '.json,.csv'
    },
    {
      key: 'cognita_results',
      name: 'Cognita 분석 결과',
      description: '관계형 데이터 분석 결과 파일 (그래프 DB)',
      accept: '.json,.csv'
    },
    {
      key: 'chronos_results',
      name: 'Chronos 분석 결과',
      description: '시계열 데이터 분석 결과 파일',
      accept: '.json,.csv'
    },
    {
      key: 'sentio_results',
      name: 'Sentio 분석 결과',
      description: '텍스트 감정 분석 결과 파일',
      accept: '.json,.csv'
    },
    {
      key: 'agora_results',
      name: 'Agora 분석 결과',
      description: '외부 시장 분석 결과 파일',
      accept: '.json,.csv'
    }
  ];

  // 분석 단계 정의
  const analysisSteps = [
    {
      title: '파일 업로드',
      description: '각 에이전트의 분석 결과 파일 업로드',
      icon: <UploadOutlined />
    },
    {
      title: '데이터 통합',
      description: '업로드된 결과들을 통합 처리',
      icon: <ApiOutlined />
    },
    {
      title: '임계값 계산',
      description: '통합된 데이터의 최적 임계값 계산',
      icon: <BarChartOutlined />
    },
    {
      title: '가중치 최적화',
      description: '각 에이전트 결과의 가중치 최적화',
      icon: <CheckCircleOutlined />
    }
  ];

  // 파일 업로드 핸들러
  const handleFileUpload = async (fileType, file) => {
    try {
      setLoading(true);
      
      // 파일 업로드 API 호출
      const response = await apiService.uploadFile(file);
      
      setUploadedFiles(prev => ({
        ...prev,
        [fileType]: {
          file: file,
          response: response,
          status: 'success'
        }
      }));

      message.success(`${requiredFiles.find(f => f.key === fileType)?.name} 업로드 완료`);
      
      // 모든 파일이 업로드되었는지 확인
      const totalFiles = requiredFiles.length;
      const uploadedCount = Object.keys(uploadedFiles).length + 1;
      
      if (uploadedCount === totalFiles) {
        setCurrentStep(1);
        await startIntegrationAnalysis();
      }
      
    } catch (error) {
      message.error(`파일 업로드 실패: ${error.message}`);
      setUploadedFiles(prev => ({
        ...prev,
        [fileType]: {
          file: file,
          status: 'error',
          error: error.message
        }
      }));
    } finally {
      setLoading(false);
    }
  };

  // 통합 분석 시작
  const startIntegrationAnalysis = async () => {
    try {
      setLoading(true);
      setCurrentStep(1);
      setProgress(25);

      // 데이터 통합 API 호출
      const integrationResponse = await apiService.integrateResults(uploadedFiles);
      setProgress(50);
      setCurrentStep(2);

      // 임계값 계산
      const thresholdResponse = await apiService.calculateThresholds();
      setProgress(75);
      setCurrentStep(3);

      // 가중치 최적화
      const weightResponse = await apiService.optimizeWeights('bayesian');
      setProgress(100);
      setCurrentStep(4);

      // 결과 설정
      const finalResults = {
        integration: integrationResponse,
        thresholds: thresholdResponse,
        weights: weightResponse,
        timestamp: new Date().toISOString()
      };

      setAnalysisResults(finalResults);
      
      // 리포트 생성
      await generateReport(finalResults);
      
      onIntegrationCompleted(finalResults);
      
    } catch (error) {
      message.error(`통합 분석 실패: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 리포트 생성
  const generateReport = async (results) => {
    try {
      const reportResponse = await apiService.generateReport({
        type: 'integration',
        data: results,
        format: 'detailed'
      });
      
      setReportData(reportResponse);
    } catch (error) {
      console.error('리포트 생성 실패:', error);
    }
  };

  // 파일 업로드 컴포넌트
  const renderFileUpload = (fileConfig) => {
    const uploadedFile = uploadedFiles[fileConfig.key];
    
    return (
      <Card
        key={fileConfig.key}
        size="small"
        title={fileConfig.name}
        extra={
          uploadedFile?.status === 'success' ? (
            <Tag color="success" icon={<CheckCircleOutlined />}>
              업로드 완료
            </Tag>
          ) : uploadedFile?.status === 'error' ? (
            <Tag color="error" icon={<ExclamationCircleOutlined />}>
              업로드 실패
            </Tag>
          ) : null
        }
        style={{ marginBottom: 16 }}
      >
        <Paragraph type="secondary" style={{ marginBottom: 16 }}>
          {fileConfig.description}
        </Paragraph>
        
        <Dragger
          accept={fileConfig.accept}
          beforeUpload={(file) => {
            handleFileUpload(fileConfig.key, file);
            return false;
          }}
          showUploadList={false}
          disabled={uploadedFile?.status === 'success'}
        >
          <p className="ant-upload-drag-icon">
            <FileTextOutlined />
          </p>
          <p className="ant-upload-text">
            {uploadedFile?.status === 'success' 
              ? `${uploadedFile.file.name} 업로드됨`
              : '파일을 드래그하거나 클릭하여 업로드'
            }
          </p>
          <p className="ant-upload-hint">
            지원 형식: {fileConfig.accept}
          </p>
        </Dragger>
        
        {uploadedFile?.status === 'error' && (
          <Alert
            message="업로드 실패"
            description={uploadedFile.error}
            type="error"
            style={{ marginTop: 8 }}
          />
        )}
      </Card>
    );
  };

  // 분석 결과 테이블
  const renderResultsTable = () => {
    if (!analysisResults) return null;

    const columns = [
      {
        title: '지표',
        dataIndex: 'metric',
        key: 'metric',
      },
      {
        title: '값',
        dataIndex: 'value',
        key: 'value',
        render: (value) => (
          <Text strong>{typeof value === 'number' ? value.toFixed(4) : value}</Text>
        )
      },
      {
        title: '상태',
        dataIndex: 'status',
        key: 'status',
        render: (status) => (
          <Tag color={status === 'optimal' ? 'success' : 'warning'}>
            {status === 'optimal' ? '최적' : '개선 필요'}
          </Tag>
        )
      }
    ];

    const data = [
      {
        key: '1',
        metric: '통합 정확도',
        value: analysisResults.integration?.accuracy || 0,
        status: (analysisResults.integration?.accuracy || 0) > 0.8 ? 'optimal' : 'needs_improvement'
      },
      {
        key: '2',
        metric: 'F1 Score',
        value: analysisResults.integration?.f1_score || 0,
        status: (analysisResults.integration?.f1_score || 0) > 0.8 ? 'optimal' : 'needs_improvement'
      },
      {
        key: '3',
        metric: '최적 임계값',
        value: analysisResults.thresholds?.optimal_threshold || 0,
        status: 'optimal'
      },
      {
        key: '4',
        metric: '가중치 최적화 점수',
        value: analysisResults.weights?.optimization_score || 0,
        status: (analysisResults.weights?.optimization_score || 0) > 0.8 ? 'optimal' : 'needs_improvement'
      }
    ];

    return (
      <Table
        columns={columns}
        dataSource={data}
        pagination={false}
        size="small"
      />
    );
  };

  return (
    <div style={{ padding: '24px' }}>
      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card>
            <Title level={2}>
              <ApiOutlined style={{ marginRight: 8 }} />
              Integration 분석
            </Title>
            <Paragraph>
              각 에이전트의 분석 결과를 통합하여 최종 예측 모델을 생성합니다.
              모든 파일을 업로드하면 자동으로 통합 분석이 시작됩니다.
            </Paragraph>
          </Card>
        </Col>

        <Col span={24}>
          <Card title="분석 진행 상황">
            <Steps current={currentStep} items={analysisSteps} />
            {progress > 0 && (
              <div style={{ marginTop: 16 }}>
                <Progress 
                  percent={progress} 
                  status={loading ? 'active' : 'success'}
                  strokeColor={{
                    '0%': '#108ee9',
                    '100%': '#87d068',
                  }}
                />
              </div>
            )}
          </Card>
        </Col>

        {currentStep === 0 && (
          <Col span={24}>
            <Card title="파일 업로드" loading={loading}>
              <Alert
                message="필수 파일 업로드"
                description="각 에이전트의 분석 결과 파일을 업로드해주세요. 모든 파일이 업로드되면 자동으로 통합 분석이 시작됩니다."
                type="info"
                style={{ marginBottom: 24 }}
              />
              
              <Row gutter={[16, 16]}>
                {requiredFiles.map(fileConfig => (
                  <Col xs={24} md={12} lg={8} key={fileConfig.key}>
                    {renderFileUpload(fileConfig)}
                  </Col>
                ))}
              </Row>
            </Card>
          </Col>
        )}

        {analysisResults && (
          <>
            <Col span={24}>
              <Card title="통합 분석 결과">
                <Row gutter={[16, 16]}>
                  <Col xs={24} sm={12} md={6}>
                    <Statistic
                      title="통합 정확도"
                      value={analysisResults.integration?.accuracy || 0}
                      precision={4}
                      suffix="%"
                      valueStyle={{ color: '#3f8600' }}
                    />
                  </Col>
                  <Col xs={24} sm={12} md={6}>
                    <Statistic
                      title="F1 Score"
                      value={analysisResults.integration?.f1_score || 0}
                      precision={4}
                      valueStyle={{ color: '#cf1322' }}
                    />
                  </Col>
                  <Col xs={24} sm={12} md={6}>
                    <Statistic
                      title="처리된 파일 수"
                      value={Object.keys(uploadedFiles).length}
                      suffix="개"
                    />
                  </Col>
                  <Col xs={24} sm={12} md={6}>
                    <Statistic
                      title="분석 완료 시간"
                      value={new Date(analysisResults.timestamp).toLocaleTimeString()}
                    />
                  </Col>
                </Row>
              </Card>
            </Col>

            <Col span={24}>
              <Card title="상세 결과">
                {renderResultsTable()}
              </Card>
            </Col>
          </>
        )}

        {reportData && (
          <Col span={24}>
            <Card title="생성된 리포트">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Alert
                  message="리포트 생성 완료"
                  description="통합 분석 리포트가 성공적으로 생성되었습니다."
                  type="success"
                />
                
                <Button
                  type="primary"
                  icon={<FileTextOutlined />}
                  onClick={() => {
                    // 리포트 다운로드 로직
                    const blob = new Blob([JSON.stringify(reportData, null, 2)], {
                      type: 'application/json'
                    });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `integration_report_${new Date().toISOString().split('T')[0]}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  리포트 다운로드
                </Button>
              </Space>
            </Card>
          </Col>
        )}
      </Row>
    </div>
  );
};

export default IntegrationAnalysis;
