import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Typography,
  Row,
  Col,
  Table,
  Tag,
  Alert,
  Spin,
  Progress,
  Space,
  Divider,
  Tabs,
  Tooltip,
  Modal,
  Select,
  Switch,
  Statistic,
  Tree,
  Image,
  Empty
} from 'antd';
import {
  ExperimentOutlined,
  EyeOutlined,
  DownloadOutlined,
  InfoCircleOutlined,
  BarChartOutlined,
  NodeIndexOutlined,
  BulbOutlined,
  FileImageOutlined,
  TableOutlined,
  TreeSelect
} from '@ant-design/icons';
import { apiService } from '../services/apiService';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const XAIResults = ({
  loading,
  setLoading,
  serverStatus,
  stepStatuses,
  supervisorResults,
  onXAICompleted
}) => {
  const [xaiResults, setXAIResults] = useState(null);
  const [selectedModel, setSelectedModel] = useState('all');
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [analysisType, setAnalysisType] = useState('feature_importance');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);

  // XAI 분석 유형
  const analysisTypes = [
    {
      key: 'feature_importance',
      name: '특성 중요도',
      description: '각 특성이 예측에 미치는 영향도 분석'
    },
    {
      key: 'shap_values',
      name: 'SHAP 값',
      description: 'SHAP을 통한 개별 예측 설명'
    },
    {
      key: 'lime_explanation',
      name: 'LIME 설명',
      description: 'LIME을 통한 지역적 설명'
    },
    {
      key: 'decision_tree',
      name: '의사결정 트리',
      description: '의사결정 과정의 시각적 표현'
    },
    {
      key: 'counterfactual',
      name: '반사실적 설명',
      description: '다른 결과를 위한 최소 변경사항'
    }
  ];

  // XAI 분석 시작
  const startXAIAnalysis = async () => {
    try {
      setIsAnalyzing(true);
      setLoading(true);
      setProgress(0);

      // 1단계: 특성 중요도 분석
      setProgress(20);
      const featureImportanceResponse = await apiService.analyzeFeatureImportance({
        supervisorResults,
        model: selectedModel
      });

      // 2단계: SHAP 분석
      setProgress(40);
      const shapResponse = await apiService.analyzeSHAP({
        supervisorResults,
        model: selectedModel
      });

      // 3단계: LIME 분석
      setProgress(60);
      const limeResponse = await apiService.analyzeLIME({
        supervisorResults,
        model: selectedModel
      });

      // 4단계: 의사결정 트리 생성
      setProgress(80);
      const decisionTreeResponse = await apiService.generateDecisionTree({
        supervisorResults,
        model: selectedModel
      });

      // 5단계: 반사실적 설명 생성
      setProgress(100);
      const counterfactualResponse = await apiService.generateCounterfactuals({
        supervisorResults,
        model: selectedModel
      });

      // 결과 통합
      const combinedResults = {
        featureImportance: featureImportanceResponse,
        shap: shapResponse,
        lime: limeResponse,
        decisionTree: decisionTreeResponse,
        counterfactual: counterfactualResponse,
        timestamp: new Date().toISOString(),
        model: selectedModel
      };

      setXAIResults(combinedResults);
      onXAICompleted(combinedResults);

    } catch (error) {
      console.error('XAI 분석 실패:', error);
      Modal.error({
        title: 'XAI 분석 실패',
        content: error.message
      });
    } finally {
      setIsAnalyzing(false);
      setLoading(false);
    }
  };

  // 특성 중요도 테이블 컬럼
  const featureImportanceColumns = [
    {
      title: '순위',
      dataIndex: 'rank',
      key: 'rank',
      width: 60,
      render: (rank) => (
        <Tag color={rank <= 3 ? 'gold' : rank <= 10 ? 'blue' : 'default'}>
          {rank}
        </Tag>
      )
    },
    {
      title: '특성명',
      dataIndex: 'feature',
      key: 'feature',
      render: (feature) => <Text strong>{feature}</Text>
    },
    {
      title: '중요도',
      dataIndex: 'importance',
      key: 'importance',
      render: (importance) => (
        <div>
          <Progress 
            percent={importance * 100} 
            size="small" 
            format={(percent) => `${percent.toFixed(2)}%`}
          />
        </div>
      )
    },
    {
      title: '설명',
      dataIndex: 'description',
      key: 'description',
      render: (description) => (
        <Tooltip title={description}>
          <Text ellipsis style={{ maxWidth: 200 }}>
            {description}
          </Text>
        </Tooltip>
      )
    },
    {
      title: '액션',
      key: 'action',
      render: (_, record) => (
        <Button
          size="small"
          icon={<EyeOutlined />}
          onClick={() => {
            setSelectedFeature(record);
            setShowDetailModal(true);
          }}
        >
          상세보기
        </Button>
      )
    }
  ];

  // SHAP 값 시각화 컴포넌트
  const SHAPVisualization = ({ data }) => {
    if (!data || !data.values) {
      return <Empty description="SHAP 데이터가 없습니다." />;
    }

    return (
      <div>
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Card title="SHAP Summary Plot" size="small">
              {data.summaryPlot ? (
                <Image
                  src={data.summaryPlot}
                  alt="SHAP Summary Plot"
                  style={{ maxWidth: '100%' }}
                />
              ) : (
                <Empty description="SHAP 요약 플롯을 생성할 수 없습니다." />
              )}
            </Card>
          </Col>
          <Col span={24}>
            <Card title="개별 예측 설명" size="small">
              <Table
                dataSource={data.individualExplanations || []}
                columns={[
                  {
                    title: '직원 ID',
                    dataIndex: 'employeeId',
                    key: 'employeeId'
                  },
                  {
                    title: '예측값',
                    dataIndex: 'prediction',
                    key: 'prediction',
                    render: (pred) => (
                      <Tag color={pred > 0.5 ? 'red' : 'green'}>
                        {pred > 0.5 ? '이직 위험' : '안전'}
                      </Tag>
                    )
                  },
                  {
                    title: '주요 기여 특성',
                    dataIndex: 'topFeatures',
                    key: 'topFeatures',
                    render: (features) => (
                      <Space wrap>
                        {features?.slice(0, 3).map((feature, idx) => (
                          <Tag key={idx} color="blue">
                            {feature.name}: {feature.value.toFixed(3)}
                          </Tag>
                        ))}
                      </Space>
                    )
                  }
                ]}
                pagination={{ pageSize: 5 }}
                size="small"
              />
            </Card>
          </Col>
        </Row>
      </div>
    );
  };

  // 의사결정 트리 컴포넌트
  const DecisionTreeVisualization = ({ data }) => {
    if (!data || !data.treeData) {
      return <Empty description="의사결정 트리 데이터가 없습니다." />;
    }

    const treeData = data.treeData.map(node => ({
      title: (
        <div>
          <Text strong>{node.feature}</Text>
          <br />
          <Text type="secondary">{node.condition}</Text>
          <br />
          <Tag color={node.prediction > 0.5 ? 'red' : 'green'}>
            {node.prediction > 0.5 ? '이직 위험' : '안전'} ({(node.confidence * 100).toFixed(1)}%)
          </Tag>
        </div>
      ),
      key: node.id,
      children: node.children
    }));

    return (
      <Card title="의사결정 트리" size="small">
        <Tree
          treeData={treeData}
          defaultExpandAll
          showLine
          showIcon={false}
        />
      </Card>
    );
  };

  // 반사실적 설명 컴포넌트
  const CounterfactualExplanation = ({ data }) => {
    if (!data || !data.explanations) {
      return <Empty description="반사실적 설명 데이터가 없습니다." />;
    }

    return (
      <div>
        <Alert
          message="반사실적 설명"
          description="현재 예측을 바꾸기 위해 필요한 최소한의 변경사항을 보여줍니다."
          type="info"
          style={{ marginBottom: 16 }}
        />
        
        {data.explanations.map((explanation, idx) => (
          <Card key={idx} title={`시나리오 ${idx + 1}`} size="small" style={{ marginBottom: 16 }}>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card title="현재 상태" size="small">
                  <Space direction="vertical">
                    {explanation.current.map((item, i) => (
                      <div key={i}>
                        <Text strong>{item.feature}:</Text> {item.value}
                      </div>
                    ))}
                  </Space>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="변경 후 상태" size="small">
                  <Space direction="vertical">
                    {explanation.counterfactual.map((item, i) => (
                      <div key={i}>
                        <Text strong>{item.feature}:</Text> 
                        <Text 
                          style={{ 
                            color: item.changed ? '#ff4d4f' : 'inherit',
                            fontWeight: item.changed ? 'bold' : 'normal'
                          }}
                        >
                          {item.value}
                        </Text>
                        {item.changed && <Tag color="red" size="small">변경</Tag>}
                      </div>
                    ))}
                  </Space>
                </Card>
              </Col>
            </Row>
            <Divider />
            <div>
              <Text strong>예측 변화:</Text> 
              <Tag color="blue">{explanation.originalPrediction}</Tag> 
              → 
              <Tag color="green">{explanation.counterfactualPrediction}</Tag>
            </div>
          </Card>
        ))}
      </div>
    );
  };

  return (
    <div style={{ padding: '24px' }}>
      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card>
            <Title level={2}>
              <ExperimentOutlined style={{ marginRight: 8 }} />
              XAI (설명 가능한 AI) 결과
            </Title>
            <Paragraph>
              AI 모델의 예측 결과를 해석하고 설명하는 다양한 기법을 통해 
              모델의 의사결정 과정을 투명하게 공개합니다.
            </Paragraph>
          </Card>
        </Col>

        <Col span={24}>
          <Card title="분석 설정">
            <Row gutter={[16, 16]} align="middle">
              <Col>
                <Text strong>분석 대상 모델:</Text>
              </Col>
              <Col>
                <Select
                  value={selectedModel}
                  onChange={setSelectedModel}
                  style={{ width: 200 }}
                  disabled={isAnalyzing}
                >
                  <Option value="all">전체 모델</Option>
                  <Option value="agora">Agora</Option>
                  <Option value="chronos">Chronos</Option>
                  <Option value="cognita">Cognita</Option>
                  <Option value="sentio">Sentio</Option>
                  <Option value="structura">Structura</Option>
                </Select>
              </Col>
              <Col>
                <Button
                  type="primary"
                  icon={<ExperimentOutlined />}
                  onClick={startXAIAnalysis}
                  disabled={!supervisorResults || isAnalyzing}
                  loading={isAnalyzing}
                >
                  XAI 분석 시작
                </Button>
              </Col>
            </Row>

            {isAnalyzing && (
              <div style={{ marginTop: 16 }}>
                <Progress 
                  percent={progress} 
                  status="active"
                  strokeColor={{
                    '0%': '#108ee9',
                    '100%': '#87d068',
                  }}
                />
                <Text type="secondary">XAI 분석을 진행하고 있습니다...</Text>
              </div>
            )}
          </Card>
        </Col>

        {!supervisorResults && (
          <Col span={24}>
            <Alert
              message="Supervisor 워크플로우 필요"
              description="XAI 분석을 시작하기 전에 Supervisor 워크플로우를 완료해주세요."
              type="warning"
              showIcon
            />
          </Col>
        )}

        {xaiResults && (
          <Col span={24}>
            <Card title="XAI 분석 결과">
              <Tabs defaultActiveKey="feature_importance">
                <TabPane 
                  tab={
                    <span>
                      <BarChartOutlined />
                      특성 중요도
                    </span>
                  } 
                  key="feature_importance"
                >
                  <Row gutter={[16, 16]}>
                    <Col span={24}>
                      <Alert
                        message="특성 중요도 분석"
                        description="모델 예측에 가장 큰 영향을 미치는 특성들을 순위별로 보여줍니다."
                        type="info"
                        style={{ marginBottom: 16 }}
                      />
                    </Col>
                    <Col span={24}>
                      <Table
                        columns={featureImportanceColumns}
                        dataSource={xaiResults.featureImportance?.features || []}
                        pagination={{ pageSize: 10 }}
                        size="small"
                      />
                    </Col>
                  </Row>
                </TabPane>

                <TabPane 
                  tab={
                    <span>
                      <NodeIndexOutlined />
                      SHAP 분석
                    </span>
                  } 
                  key="shap"
                >
                  <SHAPVisualization data={xaiResults.shap} />
                </TabPane>

                <TabPane 
                  tab={
                    <span>
                      <BulbOutlined />
                      LIME 설명
                    </span>
                  } 
                  key="lime"
                >
                  <Card title="LIME 지역적 설명" size="small">
                    {xaiResults.lime?.explanations ? (
                      <Table
                        dataSource={xaiResults.lime.explanations}
                        columns={[
                          {
                            title: '인스턴스 ID',
                            dataIndex: 'instanceId',
                            key: 'instanceId'
                          },
                          {
                            title: '예측',
                            dataIndex: 'prediction',
                            key: 'prediction',
                            render: (pred) => (
                              <Tag color={pred > 0.5 ? 'red' : 'green'}>
                                {pred > 0.5 ? '이직 위험' : '안전'}
                              </Tag>
                            )
                          },
                          {
                            title: '신뢰도',
                            dataIndex: 'confidence',
                            key: 'confidence',
                            render: (conf) => `${(conf * 100).toFixed(1)}%`
                          },
                          {
                            title: '주요 설명 특성',
                            dataIndex: 'explanationFeatures',
                            key: 'explanationFeatures',
                            render: (features) => (
                              <Space wrap>
                                {features?.slice(0, 3).map((feature, idx) => (
                                  <Tag 
                                    key={idx} 
                                    color={feature.weight > 0 ? 'red' : 'green'}
                                  >
                                    {feature.name}: {feature.weight.toFixed(3)}
                                  </Tag>
                                ))}
                              </Space>
                            )
                          }
                        ]}
                        pagination={{ pageSize: 5 }}
                        size="small"
                      />
                    ) : (
                      <Empty description="LIME 설명 데이터가 없습니다." />
                    )}
                  </Card>
                </TabPane>

                <TabPane 
                  tab={
                    <span>
                      <TreeSelect />
                      의사결정 트리
                    </span>
                  } 
                  key="decision_tree"
                >
                  <DecisionTreeVisualization data={xaiResults.decisionTree} />
                </TabPane>

                <TabPane 
                  tab={
                    <span>
                      <InfoCircleOutlined />
                      반사실적 설명
                    </span>
                  } 
                  key="counterfactual"
                >
                  <CounterfactualExplanation data={xaiResults.counterfactual} />
                </TabPane>
              </Tabs>

              <Divider />

              <Space>
                <Button
                  icon={<DownloadOutlined />}
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(xaiResults, null, 2)], {
                      type: 'application/json'
                    });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `xai_results_${new Date().toISOString().split('T')[0]}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  XAI 결과 다운로드
                </Button>
                
                <Button
                  icon={<FileImageOutlined />}
                  onClick={() => {
                    // 시각화 이미지 다운로드 로직
                    Modal.info({
                      title: '시각화 다운로드',
                      content: '시각화 이미지 다운로드 기능을 준비 중입니다.'
                    });
                  }}
                >
                  시각화 다운로드
                </Button>
              </Space>
            </Card>
          </Col>
        )}
      </Row>

      {/* 특성 상세 정보 모달 */}
      <Modal
        title="특성 상세 정보"
        open={showDetailModal}
        onCancel={() => setShowDetailModal(false)}
        footer={[
          <Button key="close" onClick={() => setShowDetailModal(false)}>
            닫기
          </Button>
        ]}
        width={600}
      >
        {selectedFeature && (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={24}>
                <Card size="small">
                  <Statistic
                    title="특성명"
                    value={selectedFeature.feature}
                  />
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <Statistic
                    title="중요도"
                    value={selectedFeature.importance}
                    precision={4}
                    suffix="%"
                  />
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <Statistic
                    title="순위"
                    value={selectedFeature.rank}
                    suffix={`/ ${xaiResults?.featureImportance?.features?.length || 0}`}
                  />
                </Card>
              </Col>
              <Col span={24}>
                <Card title="설명" size="small">
                  <Paragraph>{selectedFeature.description}</Paragraph>
                </Card>
              </Col>
            </Row>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default XAIResults;
