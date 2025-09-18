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
  Alert,
  Space,
  Divider
} from 'antd';
import {
  UploadOutlined,
  CalculatorOutlined,
  FileTextOutlined,
  BarChartOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import Papa from 'papaparse';

const { Title, Text, Paragraph } = Typography;
const { Dragger } = Upload;

const ThresholdCalculator = ({ onThresholdsCalculated, loading, setLoading, standalone = false }) => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [calculationResults, setCalculationResults] = useState(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [calculationProgress, setCalculationProgress] = useState(0);

  // F1-Score 계산 함수
  const calculateF1Score = (yTrue, yPred) => {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    
    for (let i = 0; i < yTrue.length; i++) {
      const actual = yTrue[i] === 'Yes' ? 1 : 0;
      const predicted = yPred[i];
      
      if (actual === 1 && predicted === 1) tp++;
      else if (actual === 0 && predicted === 1) fp++;
      else if (actual === 1 && predicted === 0) fn++;
      else tn++;
    }
    
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    return { f1, precision, recall, tp, fp, fn, tn };
  };

  // 최적 임계값 찾기
  const findOptimalThreshold = (scores, labels, agentName) => {
    const validIndices = scores.map((score, idx) => ({ score, label: labels[idx], idx }))
                              .filter(item => !isNaN(item.score) && item.score !== null);
    
    if (validIndices.length === 0) {
      return { threshold: 0.5, f1: 0, precision: 0, recall: 0, accuracy: 0 };
    }

    const sortedScores = validIndices.map(item => item.score).sort((a, b) => a - b);
    const minScore = Math.min(...sortedScores);
    const maxScore = Math.max(...sortedScores);
    
    // 임계값 후보 생성 (100개 구간)
    const thresholds = [];
    for (let i = 0; i <= 100; i++) {
      thresholds.push(minScore + (maxScore - minScore) * i / 100);
    }
    
    let bestThreshold = 0.5;
    let bestF1 = 0;
    let bestMetrics = null;
    
    thresholds.forEach(threshold => {
      const predictions = validIndices.map(item => item.score >= threshold ? 1 : 0);
      const actualLabels = validIndices.map(item => item.label);
      
      const metrics = calculateF1Score(actualLabels, predictions);
      
      if (metrics.f1 > bestF1) {
        bestF1 = metrics.f1;
        bestThreshold = threshold;
        bestMetrics = metrics;
      }
    });
    
    const accuracy = (bestMetrics.tp + bestMetrics.tn) / validIndices.length;
    
    return {
      threshold: bestThreshold,
      f1: bestF1,
      precision: bestMetrics.precision,
      recall: bestMetrics.recall,
      accuracy: accuracy,
      sampleSize: validIndices.length
    };
  };

  // 파일 업로드 처리
  const handleFileUpload = async (file) => {
    const isCSV = file.type === 'text/csv' || file.name.endsWith('.csv');
    if (!isCSV) {
      message.error('CSV 파일만 업로드 가능합니다.');
      return false;
    }

    try {
      setLoading(true);
      const text = await file.text();
      
      // CSV 파싱
      Papa.parse(text, {
        header: true,
        complete: (results) => {
          const data = results.data.filter(row => row.employee_id && row.attrition);
          
          if (data.length === 0) {
            message.error('유효한 데이터가 없습니다. employee_id와 attrition 컬럼이 필요합니다.');
            return;
          }

          // 필수 컬럼 확인
          const requiredColumns = ['Structura_score', 'Cognita_score', 'Chronos_score', 'Sentio_score', 'Agora_score', 'attrition'];
          const headers = Object.keys(data[0]);
          const missingColumns = requiredColumns.filter(col => !headers.includes(col));
          
          if (missingColumns.length > 0) {
            message.error(`필수 컬럼이 누락되었습니다: ${missingColumns.join(', ')}`);
            return;
          }

          setUploadedFile(file);
          message.success(`${data.length}개 행의 데이터를 로드했습니다.`);
        },
        error: (error) => {
          message.error(`파일 파싱 실패: ${error.message}`);
        }
      });
      
    } catch (error) {
      message.error(`파일 업로드 실패: ${error.message}`);
    } finally {
      setLoading(false);
    }
    
    return false; // Ant Design Upload 자동 업로드 방지
  };

  // 임계값 계산 실행
  const calculateThresholds = async () => {
    if (!uploadedFile) {
      message.error('먼저 데이터 파일을 업로드해주세요.');
      return;
    }

    try {
      setIsCalculating(true);
      setCalculationProgress(0);
      
      const text = await uploadedFile.text();
      
      Papa.parse(text, {
        header: true,
        complete: (results) => {
          const data = results.data.filter(row => row.employee_id && row.attrition);
          
          setCalculationProgress(20);
          
          // 각 에이전트별 점수 추출
          const agents = [
            { name: 'Structura', key: 'Structura_score' },
            { name: 'Cognita', key: 'Cognita_score' },
            { name: 'Chronos', key: 'Chronos_score' },
            { name: 'Sentio', key: 'Sentio_score' },
            { name: 'Agora', key: 'Agora_score' }
          ];
          
          const calculationResults = [];
          const labels = data.map(row => row.attrition);
          
          agents.forEach((agent, index) => {
            setCalculationProgress(20 + (index + 1) * 15);
            
            const scores = data.map(row => parseFloat(row[agent.key]));
            const optimal = findOptimalThreshold(scores, labels, agent.name);
            
            calculationResults.push({
              agent: agent.name,
              key: agent.key,
              ...optimal
            });
          });
          
          setCalculationProgress(100);
          setCalculationResults(calculationResults);
          
          // 부모 컴포넌트에 결과 전달
          if (onThresholdsCalculated) {
            const thresholdConfig = {};
            calculationResults.forEach(result => {
              thresholdConfig[`${result.agent.toLowerCase()}_threshold`] = result.threshold;
            });
            onThresholdsCalculated(thresholdConfig);
          }
          
          message.success('임계값 계산이 완료되었습니다!');
        }
      });
      
    } catch (error) {
      message.error(`계산 실패: ${error.message}`);
    } finally {
      setIsCalculating(false);
    }
  };

  // 결과 테이블 컬럼 정의
  const resultColumns = [
    {
      title: '에이전트',
      dataIndex: 'agent',
      key: 'agent',
      width: 120,
      render: (text) => <Text strong>{text}</Text>
    },
    {
      title: '최적 임계값',
      dataIndex: 'threshold',
      key: 'threshold',
      width: 120,
      render: (value) => <Text code>{value.toFixed(6)}</Text>
    },
    {
      title: 'F1-Score',
      dataIndex: 'f1',
      key: 'f1',
      width: 100,
      render: (value) => (
        <Text style={{ color: value > 0.7 ? '#52c41a' : value > 0.5 ? '#faad14' : '#ff4d4f' }}>
          {value.toFixed(4)}
        </Text>
      )
    },
    {
      title: '정밀도',
      dataIndex: 'precision',
      key: 'precision',
      width: 100,
      render: (value) => value.toFixed(4)
    },
    {
      title: '재현율',
      dataIndex: 'recall',
      key: 'recall',
      width: 100,
      render: (value) => value.toFixed(4)
    },
    {
      title: '정확도',
      dataIndex: 'accuracy',
      key: 'accuracy',
      width: 100,
      render: (value) => value.toFixed(4)
    },
    {
      title: '샘플 수',
      dataIndex: 'sampleSize',
      key: 'sampleSize',
      width: 100,
      render: (value) => value.toLocaleString()
    }
  ];

  return (
    <div style={{ padding: standalone ? '0' : '24px' }}>
      {!standalone && (
        <>
          <Title level={2}>
            <CalculatorOutlined /> 임계값 계산기
          </Title>
          
          <Paragraph>
            Attrition 라벨이 있는 데이터를 기반으로 각 에이전트별 최적 임계값을 계산합니다.
            F1-Score를 최대화하는 임계값을 자동으로 찾아줍니다.
          </Paragraph>
        </>
      )}

      <Alert
        message="필요한 데이터 형식"
        description="employee_id, Structura_score, Cognita_score, Chronos_score, Sentio_score, Agora_score, attrition 컬럼이 포함된 CSV 파일"
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[24, 24]}>
        {/* 파일 업로드 섹션 */}
        <Col span={24}>
          <Card title="1단계: 라벨링된 데이터 업로드" extra={<FileTextOutlined />}>
            <Dragger
              name="threshold_data"
              multiple={false}
              beforeUpload={handleFileUpload}
              showUploadList={false}
              disabled={loading || isCalculating}
            >
              <p className="ant-upload-drag-icon">
                <UploadOutlined />
              </p>
              <p className="ant-upload-text">
                Total_score.csv 또는 유사한 형식의 파일 업로드
              </p>
              <p className="ant-upload-hint">
                각 에이전트별 점수와 실제 Attrition 라벨이 포함된 CSV 파일
              </p>
            </Dragger>
            
            {uploadedFile && (
              <Alert
                message={`✅ ${uploadedFile.name} (${(uploadedFile.size/1024/1024).toFixed(2)}MB)`}
                type="success"
                showIcon
                style={{ marginTop: 16 }}
              />
            )}
          </Card>
        </Col>

        {/* 계산 실행 섹션 */}
        <Col span={24}>
          <Card title="2단계: 임계값 계산 실행" extra={<CalculatorOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                type="primary"
                size="large"
                icon={<CalculatorOutlined />}
                onClick={calculateThresholds}
                disabled={!uploadedFile || isCalculating}
                loading={isCalculating}
              >
                {isCalculating ? '계산 중...' : '최적 임계값 계산'}
              </Button>

              {isCalculating && (
                <Progress
                  percent={calculationProgress}
                  status="active"
                  strokeColor={{
                    '0%': '#108ee9',
                    '100%': '#87d068',
                  }}
                />
              )}

              <Text type="secondary">
                각 에이전트별로 F1-Score를 최대화하는 임계값을 계산합니다.
              </Text>
            </Space>
          </Card>
        </Col>

        {/* 결과 표시 섹션 */}
        {calculationResults && (
          <Col span={24}>
            <Card title="3단계: 계산 결과" extra={<BarChartOutlined />}>
              <Space direction="vertical" style={{ width: '100%' }}>
                {/* 요약 통계 */}
                <Row gutter={16}>
                  <Col span={6}>
                    <Statistic
                      title="최고 F1-Score"
                      value={Math.max(...calculationResults.map(r => r.f1))}
                      precision={4}
                      prefix={<CheckCircleOutlined />}
                      valueStyle={{ color: '#52c41a' }}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="평균 F1-Score"
                      value={calculationResults.reduce((sum, r) => sum + r.f1, 0) / calculationResults.length}
                      precision={4}
                      valueStyle={{ color: '#1890ff' }}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="최고 정확도"
                      value={Math.max(...calculationResults.map(r => r.accuracy))}
                      precision={4}
                      suffix="%"
                      formatter={(value) => (value * 100).toFixed(2)}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="총 샘플 수"
                      value={calculationResults[0]?.sampleSize || 0}
                      formatter={(value) => value.toLocaleString()}
                    />
                  </Col>
                </Row>

                <Divider />

                {/* 결과 테이블 */}
                <Table
                  columns={resultColumns}
                  dataSource={calculationResults}
                  rowKey="agent"
                  pagination={false}
                  size="middle"
                />

                <Alert
                  message="계산 완료"
                  description="위 임계값들이 배치 분석 시스템에 자동으로 적용됩니다. F1-Score가 높을수록 더 정확한 예측이 가능합니다."
                  type="success"
                  showIcon
                />
              </Space>
            </Card>
          </Col>
        )}
      </Row>
    </div>
  );
};

export default ThresholdCalculator;
