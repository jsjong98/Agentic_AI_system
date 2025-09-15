import React, { useState } from 'react';
import { 
  Card, 
  Form, 
  InputNumber, 
  Button, 
  Alert, 
  Row, 
  Col, 
  Typography, 
  Tag,
  Divider,
  Space,
  Table,
  Progress
} from 'antd';
import {
  UserOutlined,
  PredictionOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { RadialBarChart, RadialBar, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { apiService, apiUtils } from '../services/apiService';

const { Title, Text } = Typography;

const EmployeePrediction = ({ 
  thresholdResults, 
  weightResults, 
  setLoading 
}) => {
  const [form] = Form.useForm();
  const [predicting, setPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [employeeData, setEmployeeData] = useState(null);

  // 예측 실행
  const handlePredict = async (values) => {
    try {
      setPredicting(true);
      setLoading(true);
      setEmployeeData(values);
      
      const result = await apiService.predictEmployee(values);
      
      if (result.success) {
        setPredictionResult(result.predictions);
      } else {
        throw new Error(result.error || '예측에 실패했습니다.');
      }
    } catch (error) {
      message.error(`예측 실패: ${apiUtils.getErrorMessage(error)}`);
    } finally {
      setPredicting(false);
      setLoading(false);
    }
  };

  // 샘플 데이터 로드
  const loadSampleData = (type) => {
    const samples = {
      high_risk: {
        Structura_score: 0.95,
        Cognita_score: 0.6,
        Chronos_score: 0.8,
        Sentio_score: 0.7,
        Agora_score: 0.5
      },
      low_risk: {
        Structura_score: 0.1,
        Cognita_score: 0.3,
        Chronos_score: 0.0001,
        Sentio_score: 0.2,
        Agora_score: 0.15
      },
      medium_risk: {
        Structura_score: 0.5,
        Cognita_score: 0.5,
        Chronos_score: 0.1,
        Sentio_score: 0.4,
        Agora_score: 0.3
      }
    };
    
    form.setFieldsValue(samples[type]);
  };

  // 개별 Score 예측 결과 테이블
  const getThresholdPredictionColumns = () => [
    {
      title: 'Score',
      dataIndex: 'score',
      key: 'score',
      render: (text) => <Text strong>{text.replace('_score', '')}</Text>
    },
    {
      title: '입력값',
      dataIndex: 'value',
      key: 'value',
      render: (value) => <Text code>{apiUtils.formatNumber(value, 4)}</Text>
    },
    {
      title: '임계값',
      dataIndex: 'threshold',
      key: 'threshold',
      render: (value) => <Text code>{apiUtils.formatNumber(value, 4)}</Text>
    },
    {
      title: '예측',
      dataIndex: 'prediction',
      key: 'prediction',
      render: (prediction) => (
        <Tag color={prediction === '위험' ? 'red' : 'green'}>
          {prediction === '위험' ? '⚠️ 위험' : '✅ 안전'}
        </Tag>
      )
    }
  ];

  // 개별 Score 예측 데이터 준비
  const getThresholdPredictionData = () => {
    if (!predictionResult?.threshold_predictions || !employeeData) return [];
    
    const data = [];
    Object.entries(employeeData).forEach(([scoreKey, value]) => {
      const predictionKey = `${scoreKey}_prediction`;
      const thresholdKey = `${scoreKey}_threshold`;
      
      if (predictionResult.threshold_predictions[predictionKey]) {
        data.push({
          key: scoreKey,
          score: scoreKey,
          value: value,
          threshold: predictionResult.threshold_predictions[thresholdKey],
          prediction: predictionResult.threshold_predictions[predictionKey]
        });
      }
    });
    
    return data;
  };

  // 위험도 게이지 차트 데이터
  const getRiskGaugeData = () => {
    if (!predictionResult?.weighted_prediction) return [];
    
    const score = predictionResult.weighted_prediction.weighted_score;
    return [
      {
        name: '위험도',
        value: score * 100,
        fill: apiUtils.getRiskLevelColor(predictionResult.weighted_prediction.risk_level)
      }
    ];
  };

  return (
    <div>
      {/* 입력 폼 */}
      <Card title="직원 정보 입력" className="card-shadow" style={{ marginBottom: 24 }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={handlePredict}
          disabled={predicting}
        >
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12} lg={8}>
              <Form.Item
                label="Structura Score"
                name="Structura_score"
                rules={[
                  { required: true, message: 'Structura Score를 입력하세요' },
                  { type: 'number', min: 0, max: 1, message: '0과 1 사이의 값을 입력하세요' }
                ]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="0.0 ~ 1.0"
                  step={0.01}
                  precision={4}
                />
              </Form.Item>
            </Col>
            
            <Col xs={24} sm={12} lg={8}>
              <Form.Item
                label="Cognita Score"
                name="Cognita_score"
                rules={[
                  { required: true, message: 'Cognita Score를 입력하세요' },
                  { type: 'number', min: 0, max: 1, message: '0과 1 사이의 값을 입력하세요' }
                ]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="0.0 ~ 1.0"
                  step={0.01}
                  precision={4}
                />
              </Form.Item>
            </Col>
            
            <Col xs={24} sm={12} lg={8}>
              <Form.Item
                label="Chronos Score"
                name="Chronos_score"
                rules={[
                  { required: true, message: 'Chronos Score를 입력하세요' },
                  { type: 'number', min: 0, max: 1, message: '0과 1 사이의 값을 입력하세요' }
                ]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="0.0 ~ 1.0"
                  step={0.01}
                  precision={4}
                />
              </Form.Item>
            </Col>
            
            <Col xs={24} sm={12} lg={8}>
              <Form.Item
                label="Sentio Score"
                name="Sentio_score"
                rules={[
                  { required: true, message: 'Sentio Score를 입력하세요' },
                  { type: 'number', min: 0, max: 1, message: '0과 1 사이의 값을 입력하세요' }
                ]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="0.0 ~ 1.0"
                  step={0.01}
                  precision={4}
                />
              </Form.Item>
            </Col>
            
            <Col xs={24} sm={12} lg={8}>
              <Form.Item
                label="Agora Score"
                name="Agora_score"
                rules={[
                  { required: true, message: 'Agora Score를 입력하세요' },
                  { type: 'number', min: 0, max: 1, message: '0과 1 사이의 값을 입력하세요' }
                ]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="0.0 ~ 1.0"
                  step={0.01}
                  precision={4}
                />
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={[8, 8]} style={{ marginTop: 16 }}>
            <Col>
              <Button
                type="primary"
                htmlType="submit"
                icon={<PredictionOutlined />}
                loading={predicting}
                disabled={!thresholdResults || !weightResults}
              >
                예측 실행
              </Button>
            </Col>
            <Col>
              <Button onClick={() => loadSampleData('high_risk')}>
                고위험 샘플
              </Button>
            </Col>
            <Col>
              <Button onClick={() => loadSampleData('medium_risk')}>
                중위험 샘플
              </Button>
            </Col>
            <Col>
              <Button onClick={() => loadSampleData('low_risk')}>
                저위험 샘플
              </Button>
            </Col>
          </Row>
        </Form>

        {(!thresholdResults || !weightResults) && (
          <Alert
            message="모델 준비 필요"
            description="예측을 위해서는 임계값 계산과 가중치 최적화를 먼저 완료해야 합니다."
            type="warning"
            showIcon
            style={{ marginTop: 16 }}
          />
        )}
      </Card>

      {/* 예측 결과 */}
      {predictionResult && (
        <Row gutter={[16, 16]}>
          {/* 최종 예측 결과 */}
          <Col xs={24} lg={12}>
            <Card title="최종 예측 결과" className="card-shadow">
              {predictionResult.weighted_prediction && (
                <div>
                  {/* 위험도 게이지 */}
                  <div style={{ textAlign: 'center', marginBottom: 24 }}>
                    <ResponsiveContainer width="100%" height={200}>
                      <RadialBarChart
                        cx="50%"
                        cy="50%"
                        innerRadius="60%"
                        outerRadius="90%"
                        data={getRiskGaugeData()}
                        startAngle={180}
                        endAngle={0}
                      >
                        <RadialBar
                          dataKey="value"
                          cornerRadius={10}
                          fill={getRiskGaugeData()[0]?.fill}
                        />
                      </RadialBarChart>
                    </ResponsiveContainer>
                    
                    <div style={{ marginTop: -60 }}>
                      <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                        {(predictionResult.weighted_prediction.weighted_score * 100).toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '1rem', color: '#666' }}>
                        위험도 점수
                      </div>
                    </div>
                  </div>

                  {/* 예측 상세 정보 */}
                  <div style={{ 
                    padding: '20px', 
                    background: predictionResult.weighted_prediction.risk_level === '고위험군' ? '#fff2f0' :
                               predictionResult.weighted_prediction.risk_level === '주의군' ? '#fffbe6' : '#f6ffed',
                    border: `2px solid ${apiUtils.getRiskLevelColor(predictionResult.weighted_prediction.risk_level)}`,
                    borderRadius: '8px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '1.5rem', marginBottom: '8px' }}>
                      {apiUtils.getRiskLevelIcon(predictionResult.weighted_prediction.risk_level)}
                      <Text strong style={{ 
                        marginLeft: '8px',
                        color: apiUtils.getRiskLevelColor(predictionResult.weighted_prediction.risk_level)
                      }}>
                        {predictionResult.weighted_prediction.risk_level}
                      </Text>
                    </div>
                    
                    <div style={{ fontSize: '1.2rem', marginBottom: '16px' }}>
                      <Text strong>
                        최종 예측: {predictionResult.weighted_prediction.prediction_label}
                      </Text>
                    </div>
                    
                    <Row gutter={16}>
                      <Col span={12}>
                        <div>
                          <Text type="secondary">가중 점수</Text>
                          <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>
                            {apiUtils.formatNumber(predictionResult.weighted_prediction.weighted_score)}
                          </div>
                        </div>
                      </Col>
                      <Col span={12}>
                        <div>
                          <Text type="secondary">사용된 임계값</Text>
                          <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>
                            {apiUtils.formatNumber(predictionResult.weighted_prediction.threshold_used)}
                          </div>
                        </div>
                      </Col>
                    </Row>
                  </div>
                </div>
              )}
            </Card>
          </Col>

          {/* 개별 Score 예측 */}
          <Col xs={24} lg={12}>
            <Card title="개별 Score 예측" className="card-shadow">
              <Table
                columns={getThresholdPredictionColumns()}
                dataSource={getThresholdPredictionData()}
                pagination={false}
                size="small"
              />
              
              <div style={{ marginTop: 16, padding: '12px', background: '#f9f9f9', borderRadius: '6px' }}>
                <Text strong>개별 예측 요약:</Text>
                <div style={{ marginTop: 8 }}>
                  {(() => {
                    const data = getThresholdPredictionData();
                    const riskCount = data.filter(item => item.prediction === '위험').length;
                    const totalCount = data.length;
                    
                    return (
                      <div>
                        <Progress
                          percent={(riskCount / totalCount) * 100}
                          format={() => `${riskCount}/${totalCount} 위험`}
                          strokeColor={riskCount > totalCount / 2 ? '#ff4d4f' : '#52c41a'}
                        />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {totalCount}개 Score 중 {riskCount}개가 위험으로 예측됨
                        </Text>
                      </div>
                    );
                  })()}
                </div>
              </div>
            </Card>
          </Col>
        </Row>
      )}

      {/* 예측 해석 가이드 */}
      <Card title="예측 결과 해석 가이드" className="card-shadow" style={{ marginTop: 24 }}>
        <Row gutter={[16, 16]}>
          <Col xs={24} md={8}>
            <div style={{ 
              padding: '16px', 
              background: '#f6ffed', 
              border: '1px solid #b7eb8f',
              borderRadius: '8px' 
            }}>
              <Title level={5}>🟢 안전군 (0.0 ~ 0.3)</Title>
              <Text>
                Attrition 위험이 낮은 직원입니다. 
                현재 상태를 유지하며 정기적인 모니터링을 권장합니다.
              </Text>
              <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
                • 실제 이탈률: 약 1.5%<br/>
                • 권장 조치: 현상 유지, 정기 면담
              </div>
            </div>
          </Col>
          
          <Col xs={24} md={8}>
            <div style={{ 
              padding: '16px', 
              background: '#fffbe6', 
              border: '1px solid #ffe58f',
              borderRadius: '8px' 
            }}>
              <Title level={5}>🟡 주의군 (0.3 ~ 0.7)</Title>
              <Text>
                Attrition 위험이 중간 수준인 직원입니다. 
                적극적인 관리와 개입이 필요합니다.
              </Text>
              <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
                • 실제 이탈률: 약 44.3%<br/>
                • 권장 조치: 면담, 업무 조정, 복리후생 개선
              </div>
            </div>
          </Col>
          
          <Col xs={24} md={8}>
            <div style={{ 
              padding: '16px', 
              background: '#fff2f0', 
              border: '1px solid #ffccc7',
              borderRadius: '8px' 
            }}>
              <Title level={5}>🔴 고위험군 (0.7 ~ 1.0)</Title>
              <Text>
                Attrition 위험이 매우 높은 직원입니다. 
                즉시 집중적인 관리와 개입이 필요합니다.
              </Text>
              <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
                • 실제 이탈률: 약 100%<br/>
                • 권장 조치: 긴급 면담, 근무 환경 개선, 인센티브 제공
              </div>
            </div>
          </Col>
        </Row>
        
        <Divider />
        
        <Title level={5}>💡 예측 활용 방법</Title>
        <ul>
          <li><Text strong>개별 Score 분석:</Text> 어떤 영역에서 위험 신호가 나타나는지 파악</li>
          <li><Text strong>가중 점수 활용:</Text> 전체적인 위험도를 종합적으로 판단</li>
          <li><Text strong>정기적 모니터링:</Text> 주기적으로 Score를 업데이트하여 변화 추적</li>
          <li><Text strong>맞춤형 개입:</Text> 위험도 수준에 따른 차별화된 관리 전략 수립</li>
        </ul>
      </Card>
    </div>
  );
};

export default EmployeePrediction;
