import React, { useState } from 'react';
import { 
  Card, 
  Button, 
  Table, 
  Alert, 
  Row, 
  Col, 
  Statistic, 
  Progress, 
  Typography, 
  Tag,
  Divider,
  Tooltip,
  Space,
  message
} from 'antd';
import {
  CalculatorOutlined,
  TrophyOutlined,
  BarChartOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  PlayCircleOutlined
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { apiService, apiUtils } from '../services/apiService';

const { Title, Text } = Typography;

const ThresholdCalculation = ({ 
  dataLoaded, 
  thresholdResults, 
  onThresholdCalculated, 
  setLoading 
}) => {
  const [calculating, setCalculating] = useState(false);
  // eslint-disable-next-line no-unused-vars
  const [detailResults, setDetailResults] = useState(null);

  // 임계값 계산 실행
  const handleCalculateThresholds = async () => {
    if (!dataLoaded) {
      message.error('먼저 데이터를 로드해주세요.');
      return;
    }

    try {
      setCalculating(true);
      setLoading(true);
      
      const result = await apiService.calculateThresholds();
      
      if (result.success) {
        onThresholdCalculated(result.results);
        setDetailResults(result.results);
      } else {
        throw new Error(result.error || '임계값 계산에 실패했습니다.');
      }
    } catch (error) {
      message.error(`계산 실패: ${apiUtils.getErrorMessage(error)}`);
    } finally {
      setCalculating(false);
      setLoading(false);
    }
  };

  // 테이블 컬럼 정의
  const columns = [
    {
      title: 'Score',
      dataIndex: 'Score',
      key: 'Score',
      render: (text) => <Text strong>{text}</Text>,
      sorter: (a, b) => a.Score.localeCompare(b.Score),
    },
    {
      title: (
        <Space>
          최적 임계값
          <Tooltip title="F1-score가 최대가 되는 임계값">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      dataIndex: 'Optimal_Threshold',
      key: 'Optimal_Threshold',
      render: (value) => (
        <Tag color="blue" style={{ fontFamily: 'monospace' }}>
          {apiUtils.formatNumber(value, 6)}
        </Tag>
      ),
      sorter: (a, b) => a.Optimal_Threshold - b.Optimal_Threshold,
    },
    {
      title: (
        <Space>
          F1-Score
          <Tooltip title="정밀도와 재현율의 조화평균">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      dataIndex: 'F1_Score',
      key: 'F1_Score',
      render: (value) => (
        <Text style={{ 
          color: apiUtils.getPerformanceColor(value, 'f1_score'),
          fontWeight: 'bold'
        }}>
          {apiUtils.formatNumber(value)}
        </Text>
      ),
      sorter: (a, b) => b.F1_Score - a.F1_Score,
    },
    {
      title: (
        <Space>
          정밀도
          <Tooltip title="예측한 위험군 중 실제 위험군 비율">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      dataIndex: 'Precision',
      key: 'Precision',
      render: (value) => (
        <Text style={{ color: apiUtils.getPerformanceColor(value, 'precision') }}>
          {apiUtils.formatNumber(value)}
        </Text>
      ),
      sorter: (a, b) => b.Precision - a.Precision,
    },
    {
      title: (
        <Space>
          재현율
          <Tooltip title="실제 위험군 중 올바르게 예측한 비율">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      dataIndex: 'Recall',
      key: 'Recall',
      render: (value) => (
        <Text style={{ color: apiUtils.getPerformanceColor(value, 'recall') }}>
          {apiUtils.formatNumber(value)}
        </Text>
      ),
      sorter: (a, b) => b.Recall - a.Recall,
    },
    {
      title: (
        <Space>
          정확도
          <Tooltip title="전체 예측 중 올바른 예측 비율">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      dataIndex: 'Accuracy',
      key: 'Accuracy',
      render: (value) => (
        <Text style={{ color: apiUtils.getPerformanceColor(value, 'accuracy') }}>
          {apiUtils.formatNumber(value)}
        </Text>
      ),
      sorter: (a, b) => b.Accuracy - a.Accuracy,
    },
    {
      title: '등급',
      key: 'grade',
      render: (_, record) => {
        const f1 = record.F1_Score;
        let grade, color;
        
        if (f1 >= 0.8) {
          grade = 'A';
          color = 'green';
        } else if (f1 >= 0.6) {
          grade = 'B';
          color = 'orange';
        } else if (f1 >= 0.4) {
          grade = 'C';
          color = 'red';
        } else {
          grade = 'D';
          color = 'red';
        }
        
        return <Tag color={color}>{grade}</Tag>;
      },
    }
  ];

  // 성능 비교 차트 데이터 준비
  const getChartData = () => {
    if (!thresholdResults?.summary) return [];
    
    return thresholdResults.summary.map(item => ({
      name: item.Score.replace('_score', ''),
      'F1-Score': item.F1_Score,
      '정밀도': item.Precision,
      '재현율': item.Recall,
      '정확도': item.Accuracy
    }));
  };

  // 임계값 분포 차트 데이터
  const getThresholdData = () => {
    if (!thresholdResults?.summary) return [];
    
    return thresholdResults.summary.map(item => ({
      name: item.Score.replace('_score', ''),
      threshold: item.Optimal_Threshold,
      f1_score: item.F1_Score
    }));
  };

  return (
    <div>
      {/* 계산 시작 카드 */}
      <Card className="card-shadow" style={{ marginBottom: 24 }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Title level={4} style={{ margin: 0 }}>
              <CalculatorOutlined style={{ marginRight: 8 }} />
              임계값 계산
            </Title>
            <Text type="secondary">
              각 Score별 최적 임계값을 F1-score 기준으로 계산합니다.
            </Text>
          </Col>
          <Col>
            <Button
              type="primary"
              size="large"
              icon={<PlayCircleOutlined />}
              onClick={handleCalculateThresholds}
              loading={calculating}
              disabled={!dataLoaded}
            >
              {calculating ? '계산 중...' : '임계값 계산 시작'}
            </Button>
          </Col>
        </Row>

        {!dataLoaded && (
          <Alert
            message="데이터 로드 필요"
            description="임계값 계산을 위해서는 먼저 데이터를 로드해야 합니다."
            type="warning"
            showIcon
            style={{ marginTop: 16 }}
          />
        )}

        {calculating && (
          <div style={{ marginTop: 16 }}>
            <Progress percent={50} status="active" />
            <Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
              각 Score별 최적 임계값을 계산하고 있습니다...
            </Text>
          </div>
        )}
      </Card>

      {/* 계산 결과 요약 */}
      {thresholdResults && (
        <Card title="계산 결과 요약" className="card-shadow" style={{ marginBottom: 24 }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={6}>
              <Statistic
                title="계산된 Score 수"
                value={thresholdResults.summary?.length || 0}
                prefix={<BarChartOutlined />}
                suffix="개"
              />
            </Col>
            <Col xs={24} sm={6}>
              <Statistic
                title="최고 F1-Score"
                value={thresholdResults.best_score ? apiUtils.formatNumber(thresholdResults.best_score.F1_Score) : 0}
                prefix={<TrophyOutlined />}
                valueStyle={{ 
                  color: thresholdResults.best_score ? 
                    apiUtils.getPerformanceColor(thresholdResults.best_score.F1_Score, 'f1_score') : 
                    '#666'
                }}
              />
            </Col>
            <Col xs={24} sm={6}>
              <Statistic
                title="최고 성능 Score"
                value={thresholdResults.best_score?.Score?.replace('_score', '') || 'N/A'}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Col>
            <Col xs={24} sm={6}>
              <Statistic
                title="평균 F1-Score"
                value={thresholdResults.summary ? 
                  apiUtils.formatNumber(
                    thresholdResults.summary.reduce((sum, item) => sum + item.F1_Score, 0) / 
                    thresholdResults.summary.length
                  ) : 0
                }
                prefix={<BarChartOutlined />}
              />
            </Col>
          </Row>
        </Card>
      )}

      {/* 상세 결과 테이블 */}
      {thresholdResults && (
        <Card title="상세 결과" className="card-shadow" style={{ marginBottom: 24 }}>
          <Table
            columns={columns}
            dataSource={thresholdResults.summary?.map((item, index) => ({ ...item, key: index }))}
            pagination={false}
            scroll={{ x: true }}
            size="middle"
            rowClassName={(record) => 
              record.Score === thresholdResults.best_score?.Score ? 'ant-table-row-selected' : ''
            }
          />
          
          <div style={{ marginTop: 16, padding: '12px', background: '#f9f9f9', borderRadius: '6px' }}>
            <Text strong>성능 등급 기준:</Text>
            <div style={{ marginTop: 8, display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
              <span><Tag color="green">A</Tag> F1-Score ≥ 0.8 (우수)</span>
              <span><Tag color="orange">B</Tag> 0.6 ≤ F1-Score &lt; 0.8 (양호)</span>
              <span><Tag color="red">C</Tag> 0.4 ≤ F1-Score &lt; 0.6 (보통)</span>
              <span><Tag color="red">D</Tag> F1-Score &lt; 0.4 (개선 필요)</span>
            </div>
          </div>
        </Card>
      )}

      {/* 성능 비교 차트 */}
      {thresholdResults && (
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={12}>
            <Card title="성능 지표 비교" className="card-shadow">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="name" 
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis domain={[0, 1]} />
                  <RechartsTooltip formatter={(value) => [apiUtils.formatNumber(value), '']} />
                  <Legend />
                  <Bar dataKey="F1-Score" fill="#1890ff" />
                  <Bar dataKey="정밀도" fill="#52c41a" />
                  <Bar dataKey="재현율" fill="#faad14" />
                  <Bar dataKey="정확도" fill="#722ed1" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </Col>
          
          <Col xs={24} lg={12}>
            <Card title="임계값 분포" className="card-shadow">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={getThresholdData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="name" 
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
                  <RechartsTooltip 
                    formatter={(value, name) => [
                      name === 'threshold' ? apiUtils.formatNumber(value, 6) : apiUtils.formatNumber(value), 
                      name === 'threshold' ? '임계값' : 'F1-Score'
                    ]} 
                  />
                  <Legend />
                  <Bar yAxisId="left" dataKey="threshold" fill="#ff7300" name="임계값" />
                  <Line yAxisId="right" type="monotone" dataKey="f1_score" stroke="#1890ff" name="F1-Score" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>
      )}

      {/* 사용 가이드 */}
      <Card title="임계값 활용 가이드" className="card-shadow" style={{ marginTop: 24 }}>
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <Title level={5}>📊 임계값이란?</Title>
            <Text>
              각 Score에서 Attrition 위험을 판단하는 기준점입니다. 
              Score가 임계값 이상이면 '위험', 미만이면 '안전'으로 분류됩니다.
            </Text>
          </Col>
          <Col xs={24} md={12}>
            <Title level={5}>🎯 F1-Score란?</Title>
            <Text>
              정밀도와 재현율의 조화평균으로, 예측 성능을 종합적으로 평가하는 지표입니다. 
              1에 가까울수록 우수한 성능을 의미합니다.
            </Text>
          </Col>
        </Row>
        
        <Divider />
        
        <Title level={5}>💡 결과 해석 방법</Title>
        <ul>
          <li><Text strong>높은 정밀도:</Text> 위험으로 예측한 직원 중 실제 위험한 비율이 높음</li>
          <li><Text strong>높은 재현율:</Text> 실제 위험한 직원을 놓치지 않고 잘 찾아냄</li>
          <li><Text strong>높은 정확도:</Text> 전체적으로 올바른 예측을 많이 함</li>
          <li><Text strong>균형잡힌 F1-Score:</Text> 정밀도와 재현율이 모두 적절한 수준</li>
        </ul>
      </Card>
    </div>
  );
};

export default ThresholdCalculation;
