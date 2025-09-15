import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Button, Alert, Typography, Progress, Timeline } from 'antd';
import {
  ReloadOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  TrophyOutlined,
  UserOutlined,
  BarChartOutlined
} from '@ant-design/icons';
import { apiService, apiUtils } from '../services/apiService';

const { Title, Text } = Typography;

const Dashboard = ({ 
  serverStatus, 
  dataLoaded, 
  thresholdResults, 
  weightResults, 
  onRefreshStatus,
  setLoading 
}) => {
  const [systemStats, setSystemStats] = useState(null);
  const [recentActivity, setRecentActivity] = useState([]);

  useEffect(() => {
    loadSystemStats();
  }, [thresholdResults, weightResults]);

  const loadSystemStats = async () => {
    try {
      setLoading(true);
      const results = await apiService.getResults();
      setSystemStats(results.results);
      
      // 최근 활동 업데이트
      updateRecentActivity();
    } catch (error) {
      console.error('시스템 통계 로드 실패:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateRecentActivity = () => {
    const activities = [];
    
    if (weightResults) {
      activities.push({
        time: new Date().toLocaleTimeString(),
        title: '가중치 최적화 완료',
        description: `${apiUtils.getMethodName(weightResults.method)} 방법으로 F1-Score ${apiUtils.formatNumber(weightResults.best_f1_score)} 달성`,
        status: 'success',
        icon: <TrophyOutlined />
      });
    }
    
    if (thresholdResults) {
      activities.push({
        time: new Date().toLocaleTimeString(),
        title: '임계값 계산 완료',
        description: `${thresholdResults.summary?.length || 0}개 Score의 최적 임계값 계산`,
        status: 'success',
        icon: <CheckCircleOutlined />
      });
    }
    
    if (dataLoaded) {
      activities.push({
        time: new Date().toLocaleTimeString(),
        title: '데이터 로드 완료',
        description: 'HR 데이터가 성공적으로 로드됨',
        status: 'success',
        icon: <CheckCircleOutlined />
      });
    }
    
    setRecentActivity(activities);
  };

  // 전체 진행률 계산
  const calculateProgress = () => {
    let completed = 0;
    const total = 4; // 서버연결, 데이터로드, 임계값계산, 가중치최적화
    
    if (serverStatus) completed++;
    if (dataLoaded) completed++;
    if (thresholdResults) completed++;
    if (weightResults) completed++;
    
    return (completed / total) * 100;
  };

  const progress = calculateProgress();

  return (
    <div>
      {/* 시스템 상태 알림 */}
      {!serverStatus && (
        <Alert
          message="서버 연결 필요"
          description="Final_calc 백엔드 서버가 실행되지 않았습니다. 서버를 시작한 후 새로고침하세요."
          type="error"
          showIcon
          style={{ marginBottom: 24 }}
          action={
            <Button size="small" onClick={onRefreshStatus}>
              <ReloadOutlined /> 다시 확인
            </Button>
          }
        />
      )}

      {/* 전체 진행률 */}
      <Card className="card-shadow" style={{ marginBottom: 24 }}>
        <Title level={4}>시스템 설정 진행률</Title>
        <Progress 
          percent={progress} 
          status={progress === 100 ? 'success' : 'active'}
          strokeColor={{
            '0%': '#108ee9',
            '100%': '#87d068',
          }}
        />
        <div style={{ marginTop: 16, display: 'flex', justifyContent: 'space-between' }}>
          <Text type={serverStatus ? 'success' : 'secondary'}>
            {serverStatus ? '✅' : '⏳'} 서버 연결
          </Text>
          <Text type={dataLoaded ? 'success' : 'secondary'}>
            {dataLoaded ? '✅' : '⏳'} 데이터 로드
          </Text>
          <Text type={thresholdResults ? 'success' : 'secondary'}>
            {thresholdResults ? '✅' : '⏳'} 임계값 계산
          </Text>
          <Text type={weightResults ? 'success' : 'secondary'}>
            {weightResults ? '✅' : '⏳'} 가중치 최적화
          </Text>
        </div>
      </Card>

      {/* 주요 지표 카드 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card className="metric-card">
            <Statistic
              title="서버 상태"
              value={serverStatus ? '연결됨' : '연결 안됨'}
              valueStyle={{ 
                color: serverStatus ? '#3f8600' : '#cf1322',
                fontSize: '1.2rem'
              }}
              prefix={serverStatus ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card className="metric-card">
            <Statistic
              title="최고 F1-Score"
              value={weightResults ? apiUtils.formatNumber(weightResults.best_f1_score) : '미계산'}
              valueStyle={{ 
                color: weightResults ? apiUtils.getPerformanceColor(weightResults.best_f1_score, 'f1_score') : '#666',
                fontSize: '1.2rem'
              }}
              prefix={<TrophyOutlined />}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card className="metric-card">
            <Statistic
              title="계산된 임계값"
              value={thresholdResults ? `${thresholdResults.summary?.length || 0}개` : '0개'}
              valueStyle={{ 
                color: thresholdResults ? '#3f8600' : '#666',
                fontSize: '1.2rem'
              }}
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} lg={6}>
          <Card className="metric-card">
            <Statistic
              title="최적화 방법"
              value={weightResults ? apiUtils.getMethodName(weightResults.method) : '미설정'}
              valueStyle={{ 
                color: weightResults ? '#3f8600' : '#666',
                fontSize: '1rem'
              }}
              prefix={<UserOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* 임계값 결과 요약 */}
        <Col xs={24} lg={12}>
          <Card 
            title="임계값 계산 결과" 
            className="card-shadow"
            extra={
              thresholdResults && (
                <Text type="success">
                  <CheckCircleOutlined /> 계산 완료
                </Text>
              )
            }
          >
            {thresholdResults ? (
              <div>
                <div style={{ marginBottom: 16 }}>
                  <Text strong>최고 성능 Score: </Text>
                  <Text style={{ color: '#1890ff' }}>
                    {thresholdResults.best_score?.Score || 'N/A'}
                  </Text>
                  <Text type="secondary">
                    {' '}(F1-Score: {apiUtils.formatNumber(thresholdResults.best_score?.F1_Score)})
                  </Text>
                </div>
                
                <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                  {thresholdResults.summary?.slice(0, 5).map((item, index) => (
                    <div key={index} style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      padding: '8px 0',
                      borderBottom: '1px solid #f0f0f0'
                    }}>
                      <Text>{item.Score}</Text>
                      <div>
                        <Text type="secondary">임계값: </Text>
                        <Text code>{apiUtils.formatNumber(item.Optimal_Threshold)}</Text>
                        <Text type="secondary"> F1: </Text>
                        <Text style={{ color: apiUtils.getPerformanceColor(item.F1_Score, 'f1_score') }}>
                          {apiUtils.formatNumber(item.F1_Score)}
                        </Text>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#999' }}>
                <ClockCircleOutlined style={{ fontSize: '24px', marginBottom: '8px' }} />
                <div>임계값이 아직 계산되지 않았습니다.</div>
                <div style={{ fontSize: '12px' }}>데이터를 로드한 후 임계값 계산을 실행하세요.</div>
              </div>
            )}
          </Card>
        </Col>

        {/* 가중치 최적화 결과 */}
        <Col xs={24} lg={12}>
          <Card 
            title="가중치 최적화 결과" 
            className="card-shadow"
            extra={
              weightResults && (
                <Text type="success">
                  <CheckCircleOutlined /> 최적화 완료
                </Text>
              )
            }
          >
            {weightResults ? (
              <div>
                <div style={{ marginBottom: 16 }}>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Statistic
                        title="F1-Score"
                        value={apiUtils.formatNumber(weightResults.best_f1_score)}
                        valueStyle={{ 
                          color: apiUtils.getPerformanceColor(weightResults.best_f1_score, 'f1_score'),
                          fontSize: '1.1rem'
                        }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="정확도"
                        value={apiUtils.formatPercentage(weightResults.performance_metrics?.accuracy)}
                        valueStyle={{ 
                          color: apiUtils.getPerformanceColor(weightResults.performance_metrics?.accuracy, 'accuracy'),
                          fontSize: '1.1rem'
                        }}
                      />
                    </Col>
                  </Row>
                </div>

                <div>
                  <Text strong>위험도 분포:</Text>
                  <div style={{ marginTop: 8 }}>
                    {weightResults.risk_statistics?.counts && 
                      Object.entries(weightResults.risk_statistics.counts).map(([level, count]) => (
                        <div key={level} style={{ 
                          display: 'flex', 
                          justifyContent: 'space-between',
                          padding: '4px 0'
                        }}>
                          <Text>
                            {apiUtils.getRiskLevelIcon(level)} {level}
                          </Text>
                          <Text strong>{count.toLocaleString()}명</Text>
                        </div>
                      ))
                    }
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 0', color: '#999' }}>
                <ClockCircleOutlined style={{ fontSize: '24px', marginBottom: '8px' }} />
                <div>가중치가 아직 최적화되지 않았습니다.</div>
                <div style={{ fontSize: '12px' }}>임계값 계산 후 가중치 최적화를 실행하세요.</div>
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* 최근 활동 */}
      {recentActivity.length > 0 && (
        <Card title="최근 활동" className="card-shadow" style={{ marginTop: 24 }}>
          <Timeline>
            {recentActivity.map((activity, index) => (
              <Timeline.Item
                key={index}
                dot={activity.icon}
                color={activity.status === 'success' ? 'green' : 'blue'}
              >
                <div>
                  <Text strong>{activity.title}</Text>
                  <Text type="secondary" style={{ marginLeft: 8, fontSize: '12px' }}>
                    {activity.time}
                  </Text>
                </div>
                <div style={{ marginTop: 4 }}>
                  <Text type="secondary">{activity.description}</Text>
                </div>
              </Timeline.Item>
            ))}
          </Timeline>
        </Card>
      )}

      {/* 다음 단계 가이드 */}
      <Card title="다음 단계" className="card-shadow" style={{ marginTop: 24 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {!dataLoaded && (
            <Alert
              message="1단계: 데이터 업로드"
              description="HR 데이터 파일을 업로드하여 분석을 시작하세요."
              type="info"
              showIcon
            />
          )}
          
          {dataLoaded && !thresholdResults && (
            <Alert
              message="2단계: 임계값 계산"
              description="각 Score별 최적 임계값을 계산하세요."
              type="info"
              showIcon
            />
          )}
          
          {thresholdResults && !weightResults && (
            <Alert
              message="3단계: 가중치 최적화"
              description="최적의 가중치를 찾아 예측 성능을 향상시키세요."
              type="info"
              showIcon
            />
          )}
          
          {weightResults && (
            <Alert
              message="완료: 시스템 준비됨"
              description="이제 개별 직원 예측과 결과 분석을 수행할 수 있습니다."
              type="success"
              showIcon
            />
          )}
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;
