import React, { useState } from 'react';
import {
  Card,
  Button,
  Tabs,
  Typography,
  Row,
  Col,
  Alert,
  Space,
  Divider
} from 'antd';
import {
  BarChartOutlined,
  CalculatorOutlined,
  LineChartOutlined,
  PieChartOutlined,
  FileTextOutlined
} from '@ant-design/icons';
import ThresholdCalculator from './ThresholdCalculator';

const { Title, Paragraph } = Typography;
const { TabPane } = Tabs;

const PostAnalysis = ({ loading, setLoading }) => {
  const [activeTab, setActiveTab] = useState('threshold');

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <BarChartOutlined /> 사후 분석
      </Title>
      
      <Paragraph>
        이전 데이터를 기반으로 모델 성능을 분석하고 최적화 파라미터를 도출합니다.
        실제 Attrition 라벨이 있는 데이터를 사용하여 각종 분석을 수행할 수 있습니다.
      </Paragraph>

      <Alert
        message="사후 분석의 목적"
        description="과거 데이터를 통해 모델의 성능을 검증하고, 최적의 임계값과 가중치를 도출하여 향후 예측 정확도를 향상시킵니다."
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Tabs 
        activeKey={activeTab} 
        onChange={setActiveTab}
        size="large"
        type="card"
      >
        <TabPane 
          tab={
            <span>
              <CalculatorOutlined />
              임계값 최적화
            </span>
          } 
          key="threshold"
        >
          <Card>
            <ThresholdCalculator
              loading={loading}
              setLoading={setLoading}
              standalone={true}
            />
          </Card>
        </TabPane>

        <TabPane 
          tab={
            <span>
              <LineChartOutlined />
              성능 분석
            </span>
          } 
          key="performance"
        >
          <Card title="모델 성능 분석" extra={<LineChartOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message="개발 예정"
                description="각 에이전트별 성능 지표 분석, ROC 곡선, 혼동 행렬 등을 제공할 예정입니다."
                type="warning"
                showIcon
              />
              
              <Row gutter={[16, 16]}>
                <Col span={8}>
                  <Card size="small" title="ROC 곡선 분석">
                    <Paragraph type="secondary">
                      각 에이전트별 ROC 곡선을 그려 분류 성능을 시각화합니다.
                    </Paragraph>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card size="small" title="혼동 행렬">
                    <Paragraph type="secondary">
                      True Positive, False Positive 등 상세 분류 결과를 분석합니다.
                    </Paragraph>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card size="small" title="특성 중요도">
                    <Paragraph type="secondary">
                      각 특성이 예측에 미치는 영향도를 분석합니다.
                    </Paragraph>
                  </Card>
                </Col>
              </Row>
            </Space>
          </Card>
        </TabPane>

        <TabPane 
          tab={
            <span>
              <PieChartOutlined />
              가중치 최적화
            </span>
          } 
          key="weights"
        >
          <Card title="가중치 최적화" extra={<PieChartOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message="개발 예정"
                description="Bayesian Optimization, Grid Search 등을 통한 가중치 최적화 기능을 제공할 예정입니다."
                type="warning"
                showIcon
              />
              
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card size="small" title="Bayesian Optimization">
                    <Paragraph type="secondary">
                      효율적인 탐색을 통해 최적 가중치 조합을 찾습니다.
                    </Paragraph>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card size="small" title="Grid Search">
                    <Paragraph type="secondary">
                      전수 탐색을 통해 확실한 최적 가중치를 도출합니다.
                    </Paragraph>
                  </Card>
                </Col>
              </Row>
            </Space>
          </Card>
        </TabPane>

        <TabPane 
          tab={
            <span>
              <FileTextOutlined />
              비교 분석
            </span>
          } 
          key="comparison"
        >
          <Card title="모델 비교 분석" extra={<FileTextOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message="개발 예정"
                description="서로 다른 설정값들의 성능을 비교 분석하는 기능을 제공할 예정입니다."
                type="warning"
                showIcon
              />
              
              <Row gutter={[16, 16]}>
                <Col span={8}>
                  <Card size="small" title="A/B 테스트">
                    <Paragraph type="secondary">
                      두 가지 설정의 성능을 통계적으로 비교합니다.
                    </Paragraph>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card size="small" title="시계열 성능">
                    <Paragraph type="secondary">
                      시간에 따른 모델 성능 변화를 추적합니다.
                    </Paragraph>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card size="small" title="교차 검증">
                    <Paragraph type="secondary">
                      K-fold 교차 검증을 통한 안정성 평가를 수행합니다.
                    </Paragraph>
                  </Card>
                </Col>
              </Row>
            </Space>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default PostAnalysis;
