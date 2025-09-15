import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Select, Typography, Statistic, Alert } from 'antd';
import { BarChartOutlined, PieChartOutlined, LineChartOutlined } from '@ant-design/icons';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ScatterChart,
  Scatter
} from 'recharts';
import { apiUtils } from '../services/apiService';

const { Title, Text } = Typography;
const { Option } = Select;

const ResultVisualization = ({ 
  thresholdResults, 
  weightResults 
}) => {
  const [selectedChart, setSelectedChart] = useState('performance_comparison');

  // 성능 비교 차트 데이터
  const getPerformanceComparisonData = () => {
    if (!thresholdResults?.summary) return [];
    
    return thresholdResults.summary.map(item => ({
      name: item.Score.replace('_score', ''),
      'F1-Score': item.F1_Score,
      '정밀도': item.Precision,
      '재현율': item.Recall,
      '정확도': item.Accuracy
    }));
  };

  // 임계값 분포 데이터
  const getThresholdDistributionData = () => {
    if (!thresholdResults?.summary) return [];
    
    return thresholdResults.summary.map(item => ({
      name: item.Score.replace('_score', ''),
      threshold: item.Optimal_Threshold,
      f1_score: item.F1_Score
    }));
  };

  // 가중치 분포 데이터
  const getWeightDistributionData = () => {
    if (!weightResults?.optimal_weights) return [];
    
    const colors = ['#1890ff', '#52c41a', '#faad14', '#722ed1', '#f5222d'];
    
    return Object.entries(weightResults.optimal_weights).map(([key, value], index) => ({
      name: key.replace('_prediction', ''),
      value: value,
      percentage: (value * 100).toFixed(1),
      color: colors[index % colors.length]
    }));
  };

  // 위험도 분포 데이터
  const getRiskDistributionData = () => {
    if (!weightResults?.risk_statistics?.counts) return [];
    
    return Object.entries(weightResults.risk_statistics.counts).map(([level, count]) => ({
      name: level,
      count: count,
      percentage: ((count / Object.values(weightResults.risk_statistics.counts).reduce((a, b) => a + b, 0)) * 100).toFixed(1),
      color: apiUtils.getRiskLevelColor(level)
    }));
  };

  // 레이더 차트 데이터 (성능 지표)
  const getRadarData = () => {
    if (!thresholdResults?.summary) return [];
    
    return thresholdResults.summary.map(item => ({
      subject: item.Score.replace('_score', ''),
      'F1-Score': item.F1_Score * 100,
      '정밀도': item.Precision * 100,
      '재현율': item.Recall * 100,
      '정확도': item.Accuracy * 100,
      fullMark: 100
    }));
  };

  // 산점도 데이터 (정밀도 vs 재현율)
  const getScatterData = () => {
    if (!thresholdResults?.summary) return [];
    
    return thresholdResults.summary.map(item => ({
      x: item.Precision * 100,
      y: item.Recall * 100,
      name: item.Score.replace('_score', ''),
      f1: item.F1_Score
    }));
  };

  // 차트 렌더링
  const renderChart = () => {
    switch (selectedChart) {
      case 'performance_comparison':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={getPerformanceComparisonData()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis domain={[0, 1]} />
              <Tooltip formatter={(value) => [apiUtils.formatNumber(value), '']} />
              <Legend />
              <Bar dataKey="F1-Score" fill="#1890ff" />
              <Bar dataKey="정밀도" fill="#52c41a" />
              <Bar dataKey="재현율" fill="#faad14" />
              <Bar dataKey="정확도" fill="#722ed1" />
            </BarChart>
          </ResponsiveContainer>
        );

      case 'threshold_distribution':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={getThresholdDistributionData()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
              <Tooltip 
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
        );

      case 'weight_distribution':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={getWeightDistributionData()}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({name, percentage}) => `${name}: ${percentage}%`}
                outerRadius={120}
                fill="#8884d8"
                dataKey="value"
              >
                {getWeightDistributionData().map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => [`${(value * 100).toFixed(1)}%`, '가중치']} />
            </PieChart>
          </ResponsiveContainer>
        );

      case 'risk_distribution':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={getRiskDistributionData()}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({name, count, percentage}) => `${name}: ${count}명 (${percentage}%)`}
                outerRadius={120}
                fill="#8884d8"
                dataKey="count"
              >
                {getRiskDistributionData().map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => [value.toLocaleString(), '인원 수']} />
            </PieChart>
          </ResponsiveContainer>
        );

      case 'radar_performance':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart data={getRadarData()}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis domain={[0, 100]} />
              <Radar
                name="성능 지표"
                dataKey="F1-Score"
                stroke="#1890ff"
                fill="#1890ff"
                fillOpacity={0.1}
              />
              <Radar
                name="정밀도"
                dataKey="정밀도"
                stroke="#52c41a"
                fill="#52c41a"
                fillOpacity={0.1}
              />
              <Radar
                name="재현율"
                dataKey="재현율"
                stroke="#faad14"
                fill="#faad14"
                fillOpacity={0.1}
              />
              <Legend />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        );

      case 'precision_recall_scatter':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart>
              <CartesianGrid />
              <XAxis 
                type="number" 
                dataKey="x" 
                name="정밀도" 
                domain={[0, 100]}
                label={{ value: '정밀도 (%)', position: 'insideBottom', offset: -10 }}
              />
              <YAxis 
                type="number" 
                dataKey="y" 
                name="재현율" 
                domain={[0, 100]}
                label={{ value: '재현율 (%)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                formatter={(value, name) => [
                  `${value.toFixed(1)}%`, 
                  name === 'x' ? '정밀도' : '재현율'
                ]}
                labelFormatter={(label, payload) => {
                  if (payload && payload[0]) {
                    return `${payload[0].payload.name} (F1: ${apiUtils.formatNumber(payload[0].payload.f1)})`;
                  }
                  return label;
                }}
              />
              <Scatter 
                data={getScatterData()} 
                fill="#1890ff"
                shape="circle"
              />
            </ScatterChart>
          </ResponsiveContainer>
        );

      default:
        return <div>차트를 선택해주세요.</div>;
    }
  };

  // 차트 제목 및 설명
  const getChartInfo = () => {
    const chartInfo = {
      performance_comparison: {
        title: '성능 지표 비교',
        description: '각 Score별 F1-Score, 정밀도, 재현율, 정확도를 비교합니다.'
      },
      threshold_distribution: {
        title: '임계값 분포',
        description: '각 Score의 최적 임계값과 해당 F1-Score를 보여줍니다.'
      },
      weight_distribution: {
        title: '가중치 분포',
        description: '최적화된 가중치의 분포를 원형 차트로 표시합니다.'
      },
      risk_distribution: {
        title: '위험도 분포',
        description: '직원들의 위험도 구간별 분포를 보여줍니다.'
      },
      radar_performance: {
        title: '성능 지표 레이더',
        description: '여러 성능 지표를 레이더 차트로 종합 비교합니다.'
      },
      precision_recall_scatter: {
        title: '정밀도-재현율 산점도',
        description: '각 Score의 정밀도와 재현율 관계를 산점도로 표시합니다.'
      }
    };
    
    return chartInfo[selectedChart] || { title: '', description: '' };
  };

  return (
    <div>
      {/* 차트 선택 및 정보 */}
      <Card className="card-shadow" style={{ marginBottom: 24 }}>
        <Row align="middle" justify="space-between">
          <Col xs={24} lg={12}>
            <Title level={4} style={{ margin: 0 }}>
              <BarChartOutlined style={{ marginRight: 8 }} />
              결과 시각화
            </Title>
            <Text type="secondary">
              임계값 계산과 가중치 최적화 결과를 다양한 차트로 분석합니다.
            </Text>
          </Col>
          <Col xs={24} lg={12}>
            <Select
              value={selectedChart}
              onChange={setSelectedChart}
              style={{ width: '100%' }}
              placeholder="차트 유형을 선택하세요"
            >
              <Option value="performance_comparison">성능 지표 비교</Option>
              <Option value="threshold_distribution">임계값 분포</Option>
              <Option value="weight_distribution">가중치 분포</Option>
              <Option value="risk_distribution">위험도 분포</Option>
              <Option value="radar_performance">성능 레이더</Option>
              <Option value="precision_recall_scatter">정밀도-재현율 산점도</Option>
            </Select>
          </Col>
        </Row>
      </Card>

      {/* 데이터 없음 알림 */}
      {!thresholdResults && !weightResults && (
        <Alert
          message="시각화할 데이터가 없습니다"
          description="임계값 계산 또는 가중치 최적화를 먼저 수행해주세요."
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* 메인 차트 */}
      {(thresholdResults || weightResults) && (
        <Card 
          title={getChartInfo().title}
          className="card-shadow" 
          style={{ marginBottom: 24 }}
          extra={
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {getChartInfo().description}
            </Text>
          }
        >
          {renderChart()}
        </Card>
      )}

      {/* 통계 요약 */}
      {(thresholdResults || weightResults) && (
        <Row gutter={[16, 16]}>
          {/* 임계값 통계 */}
          {thresholdResults && (
            <Col xs={24} lg={12}>
              <Card title="임계값 통계" className="card-shadow">
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Statistic
                      title="최고 F1-Score"
                      value={thresholdResults.best_score ? apiUtils.formatNumber(thresholdResults.best_score.F1_Score) : 0}
                      valueStyle={{ 
                        color: thresholdResults.best_score ? 
                          apiUtils.getPerformanceColor(thresholdResults.best_score.F1_Score, 'f1_score') : 
                          '#666'
                      }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="평균 F1-Score"
                      value={thresholdResults.summary ? 
                        apiUtils.formatNumber(
                          thresholdResults.summary.reduce((sum, item) => sum + item.F1_Score, 0) / 
                          thresholdResults.summary.length
                        ) : 0
                      }
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="최고 성능 Score"
                      value={thresholdResults.best_score?.Score?.replace('_score', '') || 'N/A'}
                      valueStyle={{ color: '#1890ff' }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="계산된 Score 수"
                      value={thresholdResults.summary?.length || 0}
                      suffix="개"
                    />
                  </Col>
                </Row>
              </Card>
            </Col>
          )}

          {/* 가중치 통계 */}
          {weightResults && (
            <Col xs={24} lg={12}>
              <Card title="가중치 최적화 통계" className="card-shadow">
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Statistic
                      title="최적화된 F1-Score"
                      value={apiUtils.formatNumber(weightResults.best_f1_score)}
                      valueStyle={{ 
                        color: apiUtils.getPerformanceColor(weightResults.best_f1_score, 'f1_score'),
                        fontWeight: 'bold'
                      }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="정확도"
                      value={apiUtils.formatPercentage(weightResults.performance_metrics?.accuracy)}
                      valueStyle={{ 
                        color: apiUtils.getPerformanceColor(weightResults.performance_metrics?.accuracy, 'accuracy')
                      }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="최적화 방법"
                      value={apiUtils.getMethodName(weightResults.method)}
                      valueStyle={{ color: '#1890ff' }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="총 직원 수"
                      value={weightResults.risk_statistics?.counts ? 
                        Object.values(weightResults.risk_statistics.counts).reduce((a, b) => a + b, 0).toLocaleString() :
                        0
                      }
                      suffix="명"
                    />
                  </Col>
                </Row>
              </Card>
            </Col>
          )}
        </Row>
      )}

      {/* 차트 해석 가이드 */}
      <Card title="차트 해석 가이드" className="card-shadow" style={{ marginTop: 24 }}>
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <Title level={5}>📊 성능 지표 해석</Title>
            <ul>
              <li><Text strong>F1-Score:</Text> 정밀도와 재현율의 조화평균, 전체적인 성능 지표</li>
              <li><Text strong>정밀도:</Text> 위험으로 예측한 것 중 실제 위험한 비율</li>
              <li><Text strong>재현율:</Text> 실제 위험한 것 중 올바르게 찾아낸 비율</li>
              <li><Text strong>정확도:</Text> 전체 예측 중 올바른 예측의 비율</li>
            </ul>
          </Col>
          
          <Col xs={24} md={12}>
            <Title level={5}>🎯 차트 활용 방법</Title>
            <ul>
              <li><Text strong>성능 비교:</Text> 어떤 Score가 가장 우수한 성능을 보이는지 확인</li>
              <li><Text strong>임계값 분석:</Text> 각 Score의 최적 임계값과 성능 관계 파악</li>
              <li><Text strong>가중치 분석:</Text> 어떤 Score가 최종 예측에 가장 큰 영향을 미치는지 확인</li>
              <li><Text strong>위험도 분포:</Text> 전체 직원의 위험도 현황 파악</li>
            </ul>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default ResultVisualization;
