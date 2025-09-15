import React, { useState } from 'react';
import { 
  Card, 
  Button, 
  Select, 
  Alert, 
  Row, 
  Col, 
  Statistic, 
  Progress, 
  Typography, 
  InputNumber,
  Space,
  Table,
  Tag,
  Divider
} from 'antd';
import {
  SettingOutlined,
  PlayCircleOutlined,
  TrophyOutlined,
  CompareOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { apiService, apiUtils } from '../services/apiService';

const { Title, Text } = Typography;
const { Option } = Select;

const WeightOptimization = ({ 
  dataLoaded, 
  thresholdResults, 
  weightResults, 
  onWeightOptimized, 
  setLoading 
}) => {
  const [optimizing, setOptimizing] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState('bayesian');
  const [methodParams, setMethodParams] = useState({
    n_calls: 100,
    n_points_per_dim: 5
  });
  const [comparing, setComparing] = useState(false);
  const [comparisonResults, setComparisonResults] = useState(null);

  // 최적화 실행
  const handleOptimizeWeights = async () => {
    if (!thresholdResults) {
      message.error('먼저 임계값 계산을 완료해주세요.');
      return;
    }

    try {
      setOptimizing(true);
      setLoading(true);
      
      const params = {};
      if (selectedMethod === 'bayesian') {
        params.n_calls = methodParams.n_calls;
      } else if (selectedMethod === 'grid') {
        params.n_points_per_dim = methodParams.n_points_per_dim;
      }
      
      const result = await apiService.optimizeWeights(selectedMethod, params);
      
      if (result.success) {
        onWeightOptimized(result.results);
      } else {
        throw new Error(result.error || '가중치 최적화에 실패했습니다.');
      }
    } catch (error) {
      message.error(`최적화 실패: ${apiUtils.getErrorMessage(error)}`);
    } finally {
      setOptimizing(false);
      setLoading(false);
    }
  };

  // 방법 비교
  const handleCompareMethods = async () => {
    if (!thresholdResults) {
      message.error('먼저 임계값 계산을 완료해주세요.');
      return;
    }

    try {
      setComparing(true);
      setLoading(true);
      
      const result = await apiService.compareMethods(['grid', 'scipy']);
      
      if (result.success) {
        setComparisonResults(result);
      } else {
        throw new Error(result.error || '방법 비교에 실패했습니다.');
      }
    } catch (error) {
      message.error(`비교 실패: ${apiUtils.getErrorMessage(error)}`);
    } finally {
      setComparing(false);
      setLoading(false);
    }
  };

  // 가중치 파이 차트 데이터
  const getWeightChartData = () => {
    if (!weightResults?.optimal_weights) return [];
    
    const colors = ['#1890ff', '#52c41a', '#faad14', '#722ed1', '#f5222d'];
    
    return Object.entries(weightResults.optimal_weights).map(([key, value], index) => ({
      name: key.replace('_prediction', ''),
      value: value,
      color: colors[index % colors.length]
    }));
  };

  // 위험도 분포 차트 데이터
  const getRiskDistributionData = () => {
    if (!weightResults?.risk_statistics?.counts) return [];
    
    return Object.entries(weightResults.risk_statistics.counts).map(([level, count]) => ({
      name: level,
      count: count,
      color: apiUtils.getRiskLevelColor(level)
    }));
  };

  // 비교 결과 테이블 컬럼
  const comparisonColumns = [
    {
      title: '방법',
      dataIndex: 'method',
      key: 'method',
      render: (text) => <Text strong>{apiUtils.getMethodName(text)}</Text>
    },
    {
      title: 'F1-Score',
      dataIndex: 'best_f1_score',
      key: 'best_f1_score',
      render: (value) => (
        <Text style={{ 
          color: apiUtils.getPerformanceColor(value, 'f1_score'),
          fontWeight: 'bold'
        }}>
          {apiUtils.formatNumber(value)}
        </Text>
      ),
      sorter: (a, b) => b.best_f1_score - a.best_f1_score
    },
    {
      title: '정밀도',
      dataIndex: ['performance_metrics', 'precision'],
      key: 'precision',
      render: (value) => (
        <Text style={{ color: apiUtils.getPerformanceColor(value, 'precision') }}>
          {apiUtils.formatNumber(value)}
        </Text>
      )
    },
    {
      title: '재현율',
      dataIndex: ['performance_metrics', 'recall'],
      key: 'recall',
      render: (value) => (
        <Text style={{ color: apiUtils.getPerformanceColor(value, 'recall') }}>
          {apiUtils.formatNumber(value)}
        </Text>
      )
    },
    {
      title: '상태',
      dataIndex: 'success',
      key: 'success',
      render: (success) => (
        <Tag color={success ? 'green' : 'red'}>
          {success ? '성공' : '실패'}
        </Tag>
      )
    }
  ];

  return (
    <div>
      {/* 최적화 설정 카드 */}
      <Card className="card-shadow" style={{ marginBottom: 24 }}>
        <Row align="middle" justify="space-between">
          <Col xs={24} lg={12}>
            <Title level={4} style={{ margin: 0 }}>
              <SettingOutlined style={{ marginRight: 8 }} />
              가중치 최적화
            </Title>
            <Text type="secondary">
              여러 Score의 가중치를 최적화하여 예측 성능을 향상시킵니다.
            </Text>
          </Col>
          <Col xs={24} lg={12}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Row gutter={8}>
                <Col flex={1}>
                  <Select
                    value={selectedMethod}
                    onChange={setSelectedMethod}
                    style={{ width: '100%' }}
                    disabled={optimizing}
                  >
                    <Option value="bayesian">Bayesian Optimization</Option>
                    <Option value="grid">Grid Search</Option>
                    <Option value="scipy">Scipy Optimization</Option>
                  </Select>
                </Col>
                <Col>
                  <Button
                    type="primary"
                    icon={<PlayCircleOutlined />}
                    onClick={handleOptimizeWeights}
                    loading={optimizing}
                    disabled={!thresholdResults}
                  >
                    최적화 시작
                  </Button>
                </Col>
              </Row>
              
              {/* 방법별 파라미터 설정 */}
              {selectedMethod === 'bayesian' && (
                <div>
                  <Text>반복 횟수: </Text>
                  <InputNumber
                    min={50}
                    max={500}
                    value={methodParams.n_calls}
                    onChange={(value) => setMethodParams({...methodParams, n_calls: value})}
                    disabled={optimizing}
                  />
                </div>
              )}
              
              {selectedMethod === 'grid' && (
                <div>
                  <Text>차원당 점 수: </Text>
                  <InputNumber
                    min={3}
                    max={10}
                    value={methodParams.n_points_per_dim}
                    onChange={(value) => setMethodParams({...methodParams, n_points_per_dim: value})}
                    disabled={optimizing}
                  />
                </div>
              )}
            </Space>
          </Col>
        </Row>

        {!thresholdResults && (
          <Alert
            message="임계값 계산 필요"
            description="가중치 최적화를 위해서는 먼저 임계값 계산을 완료해야 합니다."
            type="warning"
            showIcon
            style={{ marginTop: 16 }}
          />
        )}

        {optimizing && (
          <div style={{ marginTop: 16 }}>
            <Progress percent={60} status="active" />
            <Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
              {apiUtils.getMethodName(selectedMethod)} 방법으로 최적화 중...
            </Text>
          </div>
        )}
      </Card>

      {/* 최적화 결과 요약 */}
      {weightResults && (
        <Card title="최적화 결과" className="card-shadow" style={{ marginBottom: 24 }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={6}>
              <Statistic
                title="최적화 방법"
                value={apiUtils.getMethodName(weightResults.method)}
                prefix={<SettingOutlined />}
              />
            </Col>
            <Col xs={24} sm={6}>
              <Statistic
                title="F1-Score"
                value={apiUtils.formatNumber(weightResults.best_f1_score)}
                prefix={<TrophyOutlined />}
                valueStyle={{ 
                  color: apiUtils.getPerformanceColor(weightResults.best_f1_score, 'f1_score'),
                  fontWeight: 'bold'
                }}
              />
            </Col>
            <Col xs={24} sm={6}>
              <Statistic
                title="정확도"
                value={apiUtils.formatPercentage(weightResults.performance_metrics?.accuracy)}
                valueStyle={{ 
                  color: apiUtils.getPerformanceColor(weightResults.performance_metrics?.accuracy, 'accuracy')
                }}
              />
            </Col>
            <Col xs={24} sm={6}>
              <Statistic
                title="최적 임계값"
                value={apiUtils.formatNumber(weightResults.optimal_threshold)}
                valueStyle={{ color: '#1890ff' }}
              />
            </Col>
          </Row>
        </Card>
      )}

      {/* 가중치 분포 및 위험도 분포 */}
      {weightResults && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} lg={12}>
            <Card title="최적 가중치 분포" className="card-shadow">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={getWeightChartData()}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({name, value}) => `${name}: ${(value * 100).toFixed(1)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {getWeightChartData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`${(value * 100).toFixed(1)}%`, '가중치']} />
                </PieChart>
              </ResponsiveContainer>
              
              <div style={{ marginTop: 16 }}>
                <Title level={5}>가중치 상세</Title>
                {Object.entries(weightResults.optimal_weights).map(([key, value]) => (
                  <div key={key} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    padding: '4px 0',
                    borderBottom: '1px solid #f0f0f0'
                  }}>
                    <Text>{key.replace('_prediction', '')}</Text>
                    <Text code>{apiUtils.formatNumber(value)}</Text>
                  </div>
                ))}
              </div>
            </Card>
          </Col>
          
          <Col xs={24} lg={12}>
            <Card title="위험도 분포" className="card-shadow">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getRiskDistributionData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => [value.toLocaleString(), '인원 수']} />
                  <Bar dataKey="count" fill="#1890ff" />
                </BarChart>
              </ResponsiveContainer>
              
              <div style={{ marginTop: 16 }}>
                <Title level={5}>위험도별 이탈률</Title>
                {weightResults.risk_statistics?.attrition_rates && 
                  Object.entries(weightResults.risk_statistics.attrition_rates).map(([level, rate]) => (
                    <div key={level} style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      padding: '4px 0'
                    }}>
                      <Text>
                        {apiUtils.getRiskLevelIcon(level)} {level}
                      </Text>
                      <Text strong style={{ color: apiUtils.getRiskLevelColor(level) }}>
                        {apiUtils.formatPercentage(rate)}
                      </Text>
                    </div>
                  ))
                }
              </div>
            </Card>
          </Col>
        </Row>
      )}

      {/* 방법 비교 */}
      <Card 
        title="최적화 방법 비교" 
        className="card-shadow" 
        extra={
          <Button
            icon={<CompareOutlined />}
            onClick={handleCompareMethods}
            loading={comparing}
            disabled={!thresholdResults}
          >
            방법 비교
          </Button>
        }
        style={{ marginBottom: 24 }}
      >
        {comparisonResults ? (
          <div>
            <Table
              columns={comparisonColumns}
              dataSource={comparisonResults.comparison_results?.map((item, index) => ({ ...item, key: index }))}
              pagination={false}
              size="middle"
            />
            
            {comparisonResults.best_method && (
              <Alert
                message="최고 성능 방법"
                description={
                  <div>
                    <Text strong>{apiUtils.getMethodName(comparisonResults.best_method.method)}</Text>
                    <Text> - F1-Score: {apiUtils.formatNumber(comparisonResults.best_method.best_f1_score)}</Text>
                  </div>
                }
                type="success"
                showIcon
                style={{ marginTop: 16 }}
              />
            )}
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '40px 0', color: '#999' }}>
            <CompareOutlined style={{ fontSize: '24px', marginBottom: '8px' }} />
            <div>여러 최적화 방법을 비교해보세요.</div>
            <div style={{ fontSize: '12px' }}>Grid Search와 Scipy Optimization을 비교합니다.</div>
          </div>
        )}
      </Card>

      {/* 최적화 방법 설명 */}
      <Card title="최적화 방법 설명" className="card-shadow">
        <Row gutter={[16, 16]}>
          <Col xs={24} md={8}>
            <div style={{ padding: '16px', background: '#f0f8ff', borderRadius: '8px' }}>
              <Title level={5}>🧠 Bayesian Optimization</Title>
              <Text>
                가우시안 프로세스를 사용하여 효율적으로 최적점을 탐색합니다. 
                적은 반복으로도 좋은 결과를 얻을 수 있습니다.
              </Text>
              <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
                • 권장: 정확한 결과가 필요한 경우<br/>
                • 시간: 중간 (100-200회 반복)<br/>
                • 성능: 높음
              </div>
            </div>
          </Col>
          
          <Col xs={24} md={8}>
            <div style={{ padding: '16px', background: '#f6ffed', borderRadius: '8px' }}>
              <Title level={5}>🔍 Grid Search</Title>
              <Text>
                모든 가능한 조합을 체계적으로 탐색합니다. 
                확실하지만 시간이 오래 걸릴 수 있습니다.
              </Text>
              <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
                • 권장: 빠른 테스트가 필요한 경우<br/>
                • 시간: 빠름 (3-5차원)<br/>
                • 성능: 중간
              </div>
            </div>
          </Col>
          
          <Col xs={24} md={8}>
            <div style={{ padding: '16px', background: '#fff7e6', borderRadius: '8px' }}>
              <Title level={5}>⚡ Scipy Optimization</Title>
              <Text>
                수학적 최적화 알고리즘을 사용합니다. 
                매우 빠르지만 지역 최적점에 빠질 수 있습니다.
              </Text>
              <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
                • 권장: 빠른 결과가 필요한 경우<br/>
                • 시간: 매우 빠름<br/>
                • 성능: 변동적
              </div>
            </div>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default WeightOptimization;
