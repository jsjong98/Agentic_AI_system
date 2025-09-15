import React, { useState } from 'react';
import { 
  Card, 
  Button, 
  Select, 
  Checkbox, 
  Alert, 
  Row, 
  Col, 
  Typography, 
  List,
  Tag,
  Space,
  Divider,
  message
} from 'antd';
import {
  DownloadOutlined,
  FileTextOutlined,
  FilePdfOutlined,
  FileExcelOutlined,
  CloudDownloadOutlined
} from '@ant-design/icons';
import { apiService, apiUtils } from '../services/apiService';

const { Title, Text } = Typography;
const { Option } = Select;

const ExportResults = ({ 
  thresholdResults, 
  weightResults, 
  setLoading 
}) => {
  const [exporting, setExporting] = useState(false);
  const [exportFormat, setExportFormat] = useState('csv');
  const [includeData, setIncludeData] = useState(true);
  const [exportHistory, setExportHistory] = useState([]);

  // 결과 내보내기
  const handleExport = async () => {
    if (!thresholdResults && !weightResults) {
      message.error('내보낼 결과가 없습니다. 먼저 분석을 수행해주세요.');
      return;
    }

    try {
      setExporting(true);
      setLoading(true);
      
      const result = await apiService.exportResults(exportFormat, includeData);
      
      if (result.success) {
        message.success('결과가 성공적으로 내보내졌습니다!');
        
        // 내보내기 기록 추가
        const newExport = {
          id: Date.now(),
          timestamp: new Date().toLocaleString(),
          format: exportFormat,
          includeData: includeData,
          files: result.exported_files || [],
          size: '계산 중...'
        };
        
        setExportHistory(prev => [newExport, ...prev.slice(0, 9)]); // 최근 10개만 유지
      } else {
        throw new Error(result.error || '내보내기에 실패했습니다.');
      }
    } catch (error) {
      message.error(`내보내기 실패: ${apiUtils.getErrorMessage(error)}`);
    } finally {
      setExporting(false);
      setLoading(false);
    }
  };

  // 파일 다운로드
  const handleDownload = async (filePath) => {
    try {
      setLoading(true);
      const blob = await apiService.downloadFile(filePath);
      
      // 파일 다운로드
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filePath.split('/').pop();
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      message.success('파일이 다운로드되었습니다.');
    } catch (error) {
      message.error(`다운로드 실패: ${apiUtils.getErrorMessage(error)}`);
    } finally {
      setLoading(false);
    }
  };

  // 내보내기 가능한 항목들
  const getExportableItems = () => {
    const items = [];
    
    if (thresholdResults) {
      items.push({
        title: '임계값 계산 결과',
        description: `${thresholdResults.summary?.length || 0}개 Score의 최적 임계값 및 성능 지표`,
        icon: <FileTextOutlined />,
        color: 'blue'
      });
    }
    
    if (weightResults) {
      items.push({
        title: '가중치 최적화 결과',
        description: `${apiUtils.getMethodName(weightResults.method)} 방법으로 최적화된 가중치 및 성능`,
        icon: <FileExcelOutlined />,
        color: 'green'
      });
      
      items.push({
        title: '위험도 분류 결과',
        description: `${Object.values(weightResults.risk_statistics?.counts || {}).reduce((a, b) => a + b, 0).toLocaleString()}명의 직원 위험도 분류`,
        icon: <FilePdfOutlined />,
        color: 'orange'
      });
    }
    
    return items;
  };

  // 포맷별 아이콘
  const getFormatIcon = (format) => {
    switch (format) {
      case 'csv':
        return <FileExcelOutlined style={{ color: '#52c41a' }} />;
      case 'json':
        return <FileTextOutlined style={{ color: '#1890ff' }} />;
      default:
        return <FileTextOutlined />;
    }
  };

  return (
    <div>
      {/* 내보내기 설정 */}
      <Card title="결과 내보내기" className="card-shadow" style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={12}>
            <div style={{ marginBottom: 16 }}>
              <Title level={5}>내보내기 설정</Title>
              
              <div style={{ marginBottom: 12 }}>
                <Text strong>파일 형식:</Text>
                <Select
                  value={exportFormat}
                  onChange={setExportFormat}
                  style={{ width: '100%', marginTop: 4 }}
                  disabled={exporting}
                >
                  <Option value="csv">CSV (Excel 호환)</Option>
                  <Option value="json">JSON (프로그래밍 친화적)</Option>
                </Select>
              </div>
              
              <div style={{ marginBottom: 16 }}>
                <Checkbox
                  checked={includeData}
                  onChange={(e) => setIncludeData(e.target.checked)}
                  disabled={exporting}
                >
                  <Text strong>상세 데이터 포함</Text>
                  <br />
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    개별 직원 데이터 및 예측 결과 포함
                  </Text>
                </Checkbox>
              </div>
              
              <Button
                type="primary"
                size="large"
                icon={<DownloadOutlined />}
                onClick={handleExport}
                loading={exporting}
                disabled={!thresholdResults && !weightResults}
                block
              >
                {exporting ? '내보내는 중...' : '결과 내보내기'}
              </Button>
            </div>
          </Col>
          
          <Col xs={24} lg={12}>
            <Title level={5}>내보내기 항목</Title>
            <List
              dataSource={getExportableItems()}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={item.icon}
                    title={<Text strong>{item.title}</Text>}
                    description={item.description}
                  />
                  <Tag color={item.color}>포함</Tag>
                </List.Item>
              )}
              locale={{ emptyText: '내보낼 결과가 없습니다.' }}
            />
          </Col>
        </Row>

        {(!thresholdResults && !weightResults) && (
          <Alert
            message="내보낼 데이터가 없습니다"
            description="임계값 계산 또는 가중치 최적화를 먼저 수행해주세요."
            type="warning"
            showIcon
            style={{ marginTop: 16 }}
          />
        )}
      </Card>

      {/* 내보내기 기록 */}
      {exportHistory.length > 0 && (
        <Card title="내보내기 기록" className="card-shadow" style={{ marginBottom: 24 }}>
          <List
            dataSource={exportHistory}
            renderItem={(item) => (
              <List.Item
                actions={[
                  <Button
                    type="link"
                    icon={<CloudDownloadOutlined />}
                    onClick={() => handleDownload(item.files[0])}
                    disabled={!item.files || item.files.length === 0}
                  >
                    다운로드
                  </Button>
                ]}
              >
                <List.Item.Meta
                  avatar={getFormatIcon(item.format)}
                  title={
                    <Space>
                      <Text strong>{item.format.toUpperCase()} 내보내기</Text>
                      <Tag color={item.includeData ? 'blue' : 'default'}>
                        {item.includeData ? '상세 데이터 포함' : '요약만'}
                      </Tag>
                    </Space>
                  }
                  description={
                    <div>
                      <div>{item.timestamp}</div>
                      {item.files && item.files.length > 0 && (
                        <div style={{ fontSize: '12px', color: '#666' }}>
                          {item.files.length}개 파일 생성
                        </div>
                      )}
                    </div>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      )}

      {/* 파일 형식 설명 */}
      <Card title="파일 형식 안내" className="card-shadow">
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <div style={{ 
              padding: '16px', 
              background: '#f6ffed', 
              border: '1px solid #b7eb8f',
              borderRadius: '8px' 
            }}>
              <Title level={5}>
                <FileExcelOutlined style={{ color: '#52c41a', marginRight: 8 }} />
                CSV 형식
              </Title>
              <Text>
                Excel에서 바로 열 수 있는 표 형식의 데이터입니다. 
                데이터 분석이나 보고서 작성에 적합합니다.
              </Text>
              <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
                • Excel, Google Sheets에서 직접 열기 가능<br/>
                • 표 형태로 데이터 확인 용이<br/>
                • 추가 분석 및 차트 생성 가능
              </div>
            </div>
          </Col>
          
          <Col xs={24} md={12}>
            <div style={{ 
              padding: '16px', 
              background: '#f0f8ff', 
              border: '1px solid #91d5ff',
              borderRadius: '8px' 
            }}>
              <Title level={5}>
                <FileTextOutlined style={{ color: '#1890ff', marginRight: 8 }} />
                JSON 형식
              </Title>
              <Text>
                프로그래밍에 친화적인 구조화된 데이터 형식입니다. 
                API 연동이나 자동화 시스템에 적합합니다.
              </Text>
              <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
                • 프로그래밍 언어에서 쉽게 파싱<br/>
                • 중첩된 데이터 구조 표현 가능<br/>
                • API 연동 및 자동화에 적합
              </div>
            </div>
          </Col>
        </Row>
        
        <Divider />
        
        <Title level={5}>💡 활용 방법</Title>
        <Row gutter={[16, 16]}>
          <Col xs={24} md={8}>
            <div>
              <Text strong>📊 데이터 분석</Text>
              <div style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                CSV 파일을 Excel이나 Python/R로 불러와서 추가 분석 수행
              </div>
            </div>
          </Col>
          <Col xs={24} md={8}>
            <div>
              <Text strong>📋 보고서 작성</Text>
              <div style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                결과 데이터를 PowerPoint나 Word 문서에 삽입하여 보고서 작성
              </div>
            </div>
          </Col>
          <Col xs={24} md={8}>
            <div>
              <Text strong>🔄 시스템 연동</Text>
              <div style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                JSON 데이터를 다른 시스템이나 데이터베이스로 자동 전송
              </div>
            </div>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default ExportResults;
