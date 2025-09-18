import React, { useState, useRef } from 'react';
import {
  Card,
  Button,
  Row,
  Col,
  Typography,
  Statistic,
  Table,
  Tag,
  Alert,
  Space,
  Select,
  Input,
  Divider,
  Progress,
  message,
  Spin
} from 'antd';
import {
  TeamOutlined,
  NodeIndexOutlined,
  ShareAltOutlined,
  SearchOutlined,
  ReloadOutlined,
  DownloadOutlined,
  BarChartOutlined,
  UserOutlined,
  ClusterOutlined,
  DatabaseOutlined
} from '@ant-design/icons';
import * as d3 from 'd3';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

const RelationshipAnalysis = ({ 
  loading, 
  setLoading,
  batchResults = null 
}) => {
  const [networkData, setNetworkData] = useState(null);
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [analysisType, setAnalysisType] = useState('collaboration');
  const [searchTerm, setSearchTerm] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [neo4jConnected, setNeo4jConnected] = useState(false);
  const [neo4jConfig, setNeo4jConfig] = useState({
    uri: 'bolt://44.212.67.74:7687',
    username: 'neo4j',
    password: 'legs-augmentations-cradle'
  });
  const svgRef = useRef();

  // Neo4j 연결 테스트
  const testNeo4jConnection = async () => {
    if (!neo4jConfig.uri || !neo4jConfig.username || !neo4jConfig.password) {
      message.error('Neo4j 연결 정보를 모두 입력해주세요.');
      return;
    }

    setIsAnalyzing(true);
    try {
      const response = await fetch('/api/cognita/setup/neo4j', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(neo4jConfig)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('Neo4j 연결 테스트 응답:', result);
      
      if (result.success) {
        setNeo4jConnected(true);
        message.success('Neo4j 연결 성공!');
      } else {
        setNeo4jConnected(false);
        const errorMsg = result.error || result.message || '알 수 없는 오류';
        message.error(`Neo4j 연결 실패: ${errorMsg}`);
        console.error('Neo4j 연결 실패 상세:', result);
      }
    } catch (error) {
      console.error('Neo4j 연결 테스트 실패:', error);
      setNeo4jConnected(false);
      message.error(`Neo4j 연결 테스트 실패: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Neo4j에서 직접 네트워크 분석 실행
  const analyzeRelationships = async () => {
    if (!neo4jConnected) {
      message.error('먼저 Neo4j 연결을 확인해주세요.');
      return;
    }

    setIsAnalyzing(true);
    try {
      const response = await fetch('/api/cognita/network-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysis_type: analysisType,
          search_term: searchTerm,
          neo4j_config: neo4jConfig
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setNetworkData(result.network_data);
        drawNetworkGraph(result.network_data);
        message.success('관계 분석이 완료되었습니다.');
      } else {
        throw new Error(result.error || '관계 분석에 실패했습니다.');
      }
    } catch (error) {
      console.error('관계 분석 실패:', error);
      message.error(`관계 분석 실패: ${error.message}`);
      
      // 샘플 데이터로 대체 (개발용)
      generateSampleNetworkData();
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 샘플 네트워크 데이터 생성 (개발용)
  const generateSampleNetworkData = () => {
    // 배치 분석 결과가 있으면 사용, 없으면 기본 샘플 생성
    const employees = batchResults?.results?.slice(0, 20) || 
      Array.from({length: 15}, (_, index) => ({
        employee_number: `EMP${String(index + 1).padStart(3, '0')}`,
        analysis_result: {
          combined_analysis: {
            integrated_assessment: {
              overall_risk_score: Math.random()
            }
          }
        }
      }));

    const nodes = employees.map((emp, index) => ({
      id: emp.employee_number || `emp_${index}`,
      name: `직원 ${emp.employee_number || index}`,
      department: ['HR', 'IT', 'Sales', 'Marketing', 'Finance'][index % 5],
      risk_level: emp.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || Math.random(),
      centrality: Math.random(),
      influence_score: Math.random()
    }));

    const links = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if (Math.random() > 0.7) { // 30% 확률로 연결
          links.push({
            source: nodes[i].id,
            target: nodes[j].id,
            strength: Math.random(),
            collaboration_type: ['email', 'meeting', 'project', 'mentoring'][Math.floor(Math.random() * 4)],
            frequency: Math.floor(Math.random() * 50) + 1
          });
        }
      }
    }

    const sampleData = {
      nodes,
      links,
      metrics: {
        total_employees: nodes.length,
        total_connections: links.length,
        avg_connections: (links.length * 2 / nodes.length).toFixed(1),
        network_density: (links.length / (nodes.length * (nodes.length - 1) / 2)).toFixed(3),
        clusters: Math.ceil(nodes.length / 4)
      }
    };

    setNetworkData(sampleData);
    drawNetworkGraph(sampleData);
  };

  // D3.js로 네트워크 그래프 그리기
  const drawNetworkGraph = (data) => {
    if (!data || !svgRef.current) return;

    // 기존 SVG 내용 제거
    d3.select(svgRef.current).selectAll("*").remove();

    const width = 800;
    const height = 600;
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    // 색상 스케일
    const colorScale = d3.scaleOrdinal()
      .domain(['HR', 'IT', 'Sales', 'Marketing', 'Finance'])
      .range(['#1890ff', '#52c41a', '#fa8c16', '#eb2f96', '#722ed1']);

    const riskColorScale = d3.scaleSequential()
      .domain([0, 1])
      .interpolator(d3.interpolateRdYlGn);

    // 시뮬레이션 설정
    const simulation = d3.forceSimulation(data.nodes)
      .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(25));

    // 링크 그리기
    const link = svg.append("g")
      .selectAll("line")
      .data(data.links)
      .enter().append("line")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", d => Math.sqrt(d.strength * 5));

    // 노드 그리기
    const node = svg.append("g")
      .selectAll("circle")
      .data(data.nodes)
      .enter().append("circle")
      .attr("r", d => 8 + d.centrality * 12)
      .attr("fill", d => colorScale(d.department))
      .attr("stroke", d => d3.rgb(riskColorScale(1 - d.risk_level)).darker())
      .attr("stroke-width", 3)
      .style("cursor", "pointer")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended))
      .on("click", (event, d) => {
        setSelectedEmployee(d);
      });

    // 노드 라벨
    const label = svg.append("g")
      .selectAll("text")
      .data(data.nodes)
      .enter().append("text")
      .text(d => d.name)
      .attr("font-size", "10px")
      .attr("dx", 15)
      .attr("dy", 4);

    // 시뮬레이션 업데이트
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      label
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    });

    // 드래그 함수들
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  };

  // 관계 강도 테이블 컬럼
  const relationshipColumns = [
    {
      title: '직원 A',
      dataIndex: 'source',
      key: 'source',
      render: (source) => networkData?.nodes?.find(n => n.id === source)?.name || source
    },
    {
      title: '직원 B',
      dataIndex: 'target',
      key: 'target',
      render: (target) => networkData?.nodes?.find(n => n.id === target)?.name || target
    },
    {
      title: '협업 유형',
      dataIndex: 'collaboration_type',
      key: 'collaboration_type',
      render: (type) => {
        const colors = {
          email: 'blue',
          meeting: 'green',
          project: 'orange',
          mentoring: 'purple'
        };
        return <Tag color={colors[type]}>{type}</Tag>;
      }
    },
    {
      title: '빈도',
      dataIndex: 'frequency',
      key: 'frequency',
      sorter: (a, b) => a.frequency - b.frequency,
    },
    {
      title: '관계 강도',
      dataIndex: 'strength',
      key: 'strength',
      render: (strength) => (
        <Progress 
          percent={(strength * 100).toFixed(0)} 
          size="small" 
          strokeColor={strength > 0.7 ? '#52c41a' : strength > 0.4 ? '#fa8c16' : '#ff4d4f'}
        />
      ),
      sorter: (a, b) => a.strength - b.strength,
    }
  ];

  // 중심성 분석 테이블 컬럼
  const centralityColumns = [
    {
      title: '직원명',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '부서',
      dataIndex: 'department',
      key: 'department',
      render: (dept) => <Tag color="blue">{dept}</Tag>
    },
    {
      title: '중심성',
      dataIndex: 'centrality',
      key: 'centrality',
      render: (centrality) => (
        <Progress 
          percent={(centrality * 100).toFixed(0)} 
          size="small" 
          strokeColor="#1890ff"
        />
      ),
      sorter: (a, b) => a.centrality - b.centrality,
    },
    {
      title: '영향력 점수',
      dataIndex: 'influence_score',
      key: 'influence_score',
      render: (score) => (
        <Progress 
          percent={(score * 100).toFixed(0)} 
          size="small" 
          strokeColor="#52c41a"
        />
      ),
      sorter: (a, b) => a.influence_score - b.influence_score,
    },
    {
      title: '위험도',
      dataIndex: 'risk_level',
      key: 'risk_level',
      render: (risk) => {
        const level = risk >= 0.7 ? 'HIGH' : risk >= 0.4 ? 'MEDIUM' : 'LOW';
        const color = level === 'HIGH' ? 'red' : level === 'MEDIUM' ? 'orange' : 'green';
        return <Tag color={color}>{level}</Tag>;
      }
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <TeamOutlined /> 개별 관계 분석
      </Title>
      
      <Paragraph>
        Neo4j 그래프 데이터베이스에서 직원들 간의 협업 관계, 네트워크 구조, 영향력을 분석하여 조직의 소통 패턴과 협업 효율성을 파악합니다.
        배치 분석 없이도 독립적으로 실행 가능하며, 실시간 관계 데이터를 기반으로 분석합니다.
      </Paragraph>

      {/* Neo4j 연결 설정 */}
      <Row gutter={24} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card title="Neo4j 데이터베이스 연결" extra={<DatabaseOutlined />}>
            <Row gutter={16} align="middle">
              <Col span={6}>
                <Text strong>Neo4j URI:</Text>
                <Input
                  value={neo4jConfig.uri}
                  onChange={(e) => setNeo4jConfig({...neo4jConfig, uri: e.target.value})}
                  placeholder="bolt://localhost:7687"
                  style={{ marginTop: 8 }}
                />
              </Col>
              <Col span={4}>
                <Text strong>사용자명:</Text>
                <Input
                  value={neo4jConfig.username}
                  onChange={(e) => setNeo4jConfig({...neo4jConfig, username: e.target.value})}
                  placeholder="neo4j"
                  style={{ marginTop: 8 }}
                />
              </Col>
              <Col span={4}>
                <Text strong>비밀번호:</Text>
                <Input.Password
                  value={neo4jConfig.password}
                  onChange={(e) => setNeo4jConfig({...neo4jConfig, password: e.target.value})}
                  placeholder="password"
                  style={{ marginTop: 8 }}
                />
              </Col>
              <Col span={4}>
                <Button 
                  type={neo4jConnected ? "default" : "primary"}
                  icon={<DatabaseOutlined />}
                  onClick={testNeo4jConnection}
                  loading={isAnalyzing}
                  style={{ marginTop: 24 }}
                >
                  {neo4jConnected ? '연결됨' : '연결 테스트'}
                </Button>
              </Col>
              <Col span={6}>
                {neo4jConnected && (
                  <Alert
                    message="✅ Neo4j 연결 성공"
                    type="success"
                    showIcon
                    style={{ marginTop: 8 }}
                  />
                )}
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      {/* 분석 설정 및 실행 */}
      <Row gutter={24} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card title="관계 분석 설정" extra={<NodeIndexOutlined />}>
            <Row gutter={16} align="middle">
              <Col span={6}>
                <Text strong>분석 유형:</Text>
                <Select 
                  value={analysisType} 
                  onChange={setAnalysisType}
                  style={{ width: '100%', marginTop: 8 }}
                >
                  <Option value="collaboration">협업 관계</Option>
                  <Option value="communication">소통 패턴</Option>
                  <Option value="influence">영향력 네트워크</Option>
                  <Option value="team_structure">팀 구조</Option>
                </Select>
              </Col>
              <Col span={6}>
                <Text strong>직원 검색:</Text>
                <Input
                  placeholder="직원명 또는 번호"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  prefix={<SearchOutlined />}
                  style={{ marginTop: 8 }}
                />
              </Col>
              <Col span={6}>
                <Button 
                  type="primary" 
                  icon={<ShareAltOutlined />}
                  onClick={analyzeRelationships}
                  loading={isAnalyzing}
                  disabled={!neo4jConnected}
                  style={{ marginTop: 24 }}
                >
                  관계 분석 시작
                </Button>
              </Col>
              <Col span={6}>
                <Button 
                  icon={<ReloadOutlined />}
                  onClick={() => networkData && drawNetworkGraph(networkData)}
                  disabled={!networkData}
                  style={{ marginTop: 24 }}
                >
                  그래프 새로고침
                </Button>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      {!neo4jConnected && (
        <Alert
          message="Neo4j 연결 필요"
          description="관계 분석을 위해서는 먼저 Neo4j 그래프 데이터베이스에 연결해주세요. 연결 정보를 입력하고 '연결 테스트' 버튼을 클릭하세요."
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {batchResults && (
        <Alert
          message="배치 분석 결과 연동됨"
          description="배치 분석 결과가 연동되어 더 정확한 위험도 정보를 제공합니다."
          type="success"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* 네트워크 메트릭스 */}
      {networkData && (
        <Row gutter={24} style={{ marginBottom: 24 }}>
          <Col span={24}>
            <Card title="네트워크 개요" extra={<BarChartOutlined />}>
              <Row gutter={16}>
                <Col span={4}>
                  <Statistic
                    title="총 직원 수"
                    value={networkData.metrics.total_employees}
                    prefix={<UserOutlined />}
                  />
                </Col>
                <Col span={4}>
                  <Statistic
                    title="총 연결 수"
                    value={networkData.metrics.total_connections}
                    prefix={<ShareAltOutlined />}
                  />
                </Col>
                <Col span={4}>
                  <Statistic
                    title="평균 연결 수"
                    value={networkData.metrics.avg_connections}
                    prefix={<NodeIndexOutlined />}
                  />
                </Col>
                <Col span={4}>
                  <Statistic
                    title="네트워크 밀도"
                    value={networkData.metrics.network_density}
                    prefix={<ClusterOutlined />}
                  />
                </Col>
                <Col span={4}>
                  <Statistic
                    title="클러스터 수"
                    value={networkData.metrics.clusters}
                    prefix={<TeamOutlined />}
                  />
                </Col>
                <Col span={4}>
                  <Button 
                    type="primary" 
                    icon={<DownloadOutlined />}
                    onClick={() => {
                      // 네트워크 데이터 내보내기
                      const dataStr = JSON.stringify(networkData, null, 2);
                      const dataBlob = new Blob([dataStr], {type: 'application/json'});
                      const url = URL.createObjectURL(dataBlob);
                      const link = document.createElement('a');
                      link.href = url;
                      link.download = 'network_analysis.json';
                      link.click();
                    }}
                  >
                    데이터 내보내기
                  </Button>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      )}

      <Row gutter={24}>
        {/* 네트워크 그래프 */}
        <Col span={16}>
          <Card title="협업 네트워크 그래프" extra={<NodeIndexOutlined />}>
            {isAnalyzing ? (
              <div style={{ textAlign: 'center', padding: '100px 0' }}>
                <Spin size="large" />
                <div style={{ marginTop: 16 }}>
                  <Text>네트워크 분석 중...</Text>
                </div>
              </div>
            ) : networkData ? (
              <div>
                <svg ref={svgRef} style={{ border: '1px solid #d9d9d9', borderRadius: '6px' }}></svg>
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">
                    💡 <strong>사용법:</strong> 노드를 클릭하면 상세 정보를 확인할 수 있고, 드래그하여 위치를 조정할 수 있습니다.
                  </Text>
                  <br />
                  <Text type="secondary">
                    🎨 <strong>색상:</strong> 부서별로 다른 색상, 테두리는 위험도를 나타냅니다. 노드 크기는 중심성을 반영합니다.
                  </Text>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '100px 0' }}>
                <NodeIndexOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">관계 분석을 시작하여 네트워크 그래프를 확인하세요</Text>
                </div>
              </div>
            )}
          </Card>
        </Col>

        {/* 선택된 직원 상세 정보 */}
        <Col span={8}>
          <Card title="직원 상세 정보" extra={<UserOutlined />}>
            {selectedEmployee ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>직원명:</Text> {selectedEmployee.name}
                </div>
                <div>
                  <Text strong>부서:</Text> <Tag color="blue">{selectedEmployee.department}</Tag>
                </div>
                <div>
                  <Text strong>중심성:</Text>
                  <Progress 
                    percent={(selectedEmployee.centrality * 100).toFixed(0)} 
                    size="small" 
                    strokeColor="#1890ff"
                  />
                </div>
                <div>
                  <Text strong>영향력:</Text>
                  <Progress 
                    percent={(selectedEmployee.influence_score * 100).toFixed(0)} 
                    size="small" 
                    strokeColor="#52c41a"
                  />
                </div>
                <div>
                  <Text strong>위험도:</Text>
                  <Progress 
                    percent={(selectedEmployee.risk_level * 100).toFixed(0)} 
                    size="small" 
                    strokeColor={selectedEmployee.risk_level > 0.7 ? '#ff4d4f' : selectedEmployee.risk_level > 0.4 ? '#fa8c16' : '#52c41a'}
                  />
                </div>
                <Divider />
                <div>
                  <Text strong>연결된 동료 수:</Text> {
                    networkData?.links?.filter(link => 
                      link.source === selectedEmployee.id || link.target === selectedEmployee.id
                    ).length || 0
                  }명
                </div>
                <div>
                  <Text strong>주요 협업 유형:</Text>
                  {networkData?.links?.filter(link => 
                    link.source === selectedEmployee.id || link.target === selectedEmployee.id
                  ).map(link => (
                    <Tag key={`${link.source}-${link.target}`} style={{ margin: '2px' }}>
                      {link.collaboration_type}
                    </Tag>
                  ))}
                </div>
              </Space>
            ) : (
              <div style={{ textAlign: 'center', padding: '50px 0' }}>
                <UserOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">그래프에서 직원을 선택하세요</Text>
                </div>
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* 관계 분석 테이블들 */}
      {networkData && (
        <Row gutter={24} style={{ marginTop: 24 }}>
          <Col span={12}>
            <Card title="협업 관계 상세" extra={<ShareAltOutlined />}>
              <Table
                columns={relationshipColumns}
                dataSource={networkData.links}
                rowKey={(record) => `${record.source}-${record.target}`}
                pagination={{ pageSize: 10 }}
                size="small"
              />
            </Card>
          </Col>
          <Col span={12}>
            <Card title="중심성 및 영향력 분석" extra={<BarChartOutlined />}>
              <Table
                columns={centralityColumns}
                dataSource={networkData.nodes}
                rowKey="id"
                pagination={{ pageSize: 10 }}
                size="small"
              />
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
};

export default RelationshipAnalysis;
