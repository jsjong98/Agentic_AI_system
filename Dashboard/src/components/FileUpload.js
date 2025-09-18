import React, { useState, useCallback } from 'react';
import { Card, Button, message, Alert, Table, Typography, Row, Col, Statistic, Progress, Form, Input } from 'antd';
import { 
  InboxOutlined, 
  UploadOutlined, 
  FileTextOutlined, 
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  DatabaseOutlined,
  LinkOutlined
} from '@ant-design/icons';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import { apiService, apiUtils } from '../services/apiService';

const { Title, Text } = Typography;

const FileUpload = ({ onDataLoaded, setLoading, moduleType = 'default' }) => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [dataPreview, setDataPreview] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [validationResults, setValidationResults] = useState(null);
  
  // Neo4j 연결 관련 상태 (Cognita용)
  const [neo4jConfig, setNeo4jConfig] = useState({
    uri: 'bolt://44.212.67.74:7687',
    username: 'neo4j',
    password: 'legs-augmentations-cradle'
  });
  const [neo4jConnecting, setNeo4jConnecting] = useState(false);
  const [neo4jConnected, setNeo4jConnected] = useState(false);

  // 파일 드롭 핸들러
  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      message.error('지원하지 않는 파일 형식입니다. CSV 파일만 업로드 가능합니다.');
      return;
    }

    const file = acceptedFiles[0];
    if (file) {
      handleFileSelect(file);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv'],
      'text/plain': ['.csv']
    },
    multiple: false,
    maxSize: 500 * 1024 * 1024 // 500MB
  });

  // Neo4j 연결 테스트 함수
  const handleNeo4jConnect = async () => {
    setNeo4jConnecting(true);
    try {
      const response = await fetch('/api/cognita/setup/neo4j', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(neo4jConfig)
      });

      const result = await response.json();
      
      if (result.success) {
        setNeo4jConnected(true);
        message.success('Neo4j 데이터베이스에 성공적으로 연결되었습니다!');
        
        // 연결 성공 시 데이터 로드됨을 알림
        if (onDataLoaded) {
          onDataLoaded({
            type: 'neo4j',
            config: neo4jConfig,
            connected: true,
            employees: result.employees || [],
            departments: result.departments || []
          });
        }
      } else {
        message.error(`연결 실패: ${result.error}`);
        setNeo4jConnected(false);
      }
    } catch (error) {
      console.error('Neo4j 연결 오류:', error);
      message.error('Neo4j 연결 중 오류가 발생했습니다.');
      setNeo4jConnected(false);
    } finally {
      setNeo4jConnecting(false);
    }
  };

  // 파일 선택 처리
  const handleFileSelect = (file) => {
    setUploadedFile(file);
    setUploadProgress(0);
    
    // CSV 파일 미리보기
    if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
      previewCSVFile(file);
    }
  };

  // CSV 파일 미리보기
  const previewCSVFile = (file) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      preview: 10, // 처음 10행만 미리보기
      complete: (results) => {
        if (results.errors.length > 0) {
          message.error('CSV 파일 파싱 중 오류가 발생했습니다.');
          console.error('CSV 파싱 오류:', results.errors);
          return;
        }

        setDataPreview({
          headers: results.meta.fields,
          data: results.data,
          totalRows: results.data.length
        });

        // 데이터 유효성 검사
        validateData(results.meta.fields, results.data);
      },
      error: (error) => {
        message.error('파일을 읽는 중 오류가 발생했습니다.');
        console.error('파일 읽기 오류:', error);
      }
    });
  };

  // 데이터 유효성 검사
  const validateData = (headers, data) => {
    const requiredColumns = ['attrition'];
    const scoreColumns = ['Structura_score', 'Cognita_score', 'Chronos_score', 'Sentio_score', 'Agora_score'];
    
    const validation = {
      hasRequiredColumns: true,
      missingColumns: [],
      hasScoreColumns: true,
      missingScoreColumns: [],
      dataQuality: {
        totalRows: data.length,
        validRows: 0,
        missingValues: 0,
        invalidValues: 0
      }
    };

    // 필수 컬럼 확인
    requiredColumns.forEach(col => {
      if (!headers.includes(col)) {
        validation.hasRequiredColumns = false;
        validation.missingColumns.push(col);
      }
    });

    // Score 컬럼 확인
    const foundScoreColumns = scoreColumns.filter(col => headers.includes(col));
    if (foundScoreColumns.length === 0) {
      validation.hasScoreColumns = false;
      validation.missingScoreColumns = scoreColumns;
    } else if (foundScoreColumns.length < scoreColumns.length) {
      validation.missingScoreColumns = scoreColumns.filter(col => !headers.includes(col));
    }

    // 데이터 품질 확인
    data.forEach(row => {
      let isValidRow = true;
      let missingCount = 0;
      let invalidCount = 0;

      Object.values(row).forEach(value => {
        if (value === null || value === undefined || value === '') {
          missingCount++;
        }
      });

      // Score 컬럼의 숫자 유효성 확인
      foundScoreColumns.forEach(col => {
        const value = row[col];
        if (value && isNaN(parseFloat(value))) {
          invalidCount++;
          isValidRow = false;
        }
      });

      if (isValidRow && missingCount === 0) {
        validation.dataQuality.validRows++;
      }
      validation.dataQuality.missingValues += missingCount;
      validation.dataQuality.invalidValues += invalidCount;
    });

    setValidationResults(validation);
  };

  // 파일 업로드 및 데이터 로드
  const handleUpload = async () => {
    if (!uploadedFile) {
      message.error('업로드할 파일을 선택해주세요.');
      return;
    }

    try {
      setLoading(true);
      setUploadProgress(10);

      // 임시로 파일명을 사용하여 데이터 로드 (실제로는 파일 업로드 API 필요)
      message.info('파일을 서버에 업로드하고 있습니다...');
      setUploadProgress(50);

      // 기본 데이터 파일 로드 (실제 구현에서는 업로드된 파일 사용)
      const result = await apiService.loadData(uploadedFile.name);
      setUploadProgress(100);

      if (result.success) {
        setFileData(result);
        onDataLoaded(true);
        message.success('데이터가 성공적으로 로드되었습니다!');
      } else {
        throw new Error(result.error || '데이터 로드에 실패했습니다.');
      }
    } catch (error) {
      message.error(`업로드 실패: ${apiUtils.getErrorMessage(error)}`);
      onDataLoaded(false);
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  // 기본 데이터 파일 로드
  const loadDefaultData = async () => {
    try {
      setLoading(true);
      const result = await apiService.loadData('Total_score.csv');
      
      if (result.success) {
        setFileData(result);
        onDataLoaded(true);
        message.success('기본 데이터가 성공적으로 로드되었습니다!');
      } else {
        throw new Error(result.error || '데이터 로드에 실패했습니다.');
      }
    } catch (error) {
      message.error(`데이터 로드 실패: ${apiUtils.getErrorMessage(error)}`);
      onDataLoaded(false);
    } finally {
      setLoading(false);
    }
  };

  // 테이블 컬럼 설정
  const getTableColumns = () => {
    if (!dataPreview?.headers) return [];
    
    return dataPreview.headers.slice(0, 8).map(header => ({
      title: header,
      dataIndex: header,
      key: header,
      ellipsis: true,
      width: 120,
      render: (text) => (
        <Text style={{ fontSize: '12px' }}>
          {text?.toString().substring(0, 20)}
          {text?.toString().length > 20 ? '...' : ''}
        </Text>
      )
    }));
  };

  return (
    <div>
      {/* Cognita 모듈용 Neo4j 연결 설정 */}
      {moduleType === 'cognita' ? (
        <Card title="Neo4j 데이터베이스 연결" className="card-shadow" style={{ marginBottom: 24 }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Form layout="vertical">
                <Form.Item label="Neo4j URI" required>
                  <Input
                    value={neo4jConfig.uri}
                    onChange={(e) => setNeo4jConfig({...neo4jConfig, uri: e.target.value})}
                    placeholder="bolt://localhost:7687"
                    prefix={<DatabaseOutlined />}
                  />
                </Form.Item>
                <Form.Item label="사용자명" required>
                  <Input
                    value={neo4jConfig.username}
                    onChange={(e) => setNeo4jConfig({...neo4jConfig, username: e.target.value})}
                    placeholder="neo4j"
                  />
                </Form.Item>
                <Form.Item label="비밀번호" required>
                  <Input.Password
                    value={neo4jConfig.password}
                    onChange={(e) => setNeo4jConfig({...neo4jConfig, password: e.target.value})}
                    placeholder="password"
                  />
                </Form.Item>
                <Form.Item>
                  <Button
                    type="primary"
                    icon={<LinkOutlined />}
                    onClick={handleNeo4jConnect}
                    loading={neo4jConnecting}
                    disabled={neo4jConnected}
                  >
                    {neo4jConnected ? '연결됨' : '연결 테스트'}
                  </Button>
                  {neo4jConnected && (
                    <Alert
                      message="Neo4j 데이터베이스에 성공적으로 연결되었습니다!"
                      type="success"
                      showIcon
                      style={{ marginTop: 16 }}
                    />
                  )}
                </Form.Item>
              </Form>
            </Col>
          </Row>
        </Card>
      ) : (
        /* 기본 파일 업로드 영역 */
        <Card title="데이터 파일 업로드" className="card-shadow" style={{ marginBottom: 24 }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <div {...getRootProps()} className={`upload-area ${isDragActive ? 'dragover' : ''}`}>
                <input {...getInputProps()} />
                <p className="ant-upload-drag-icon">
                  <InboxOutlined style={{ fontSize: '48px', color: '#1890ff' }} />
                </p>
                <p className="ant-upload-text">
                  {isDragActive ? '파일을 여기에 놓으세요' : '파일을 드래그하거나 클릭하여 업로드'}
                </p>
                <p className="ant-upload-hint">
                  CSV 형식의 HR 데이터 파일을 업로드하세요. (최대 500MB)
                </p>
                {uploadedFile && (
                  <div style={{ marginTop: 16, padding: '12px', background: '#f0f8ff', borderRadius: '6px' }}>
                    <FileTextOutlined style={{ color: '#1890ff', marginRight: '8px' }} />
                    <Text strong>{uploadedFile.name}</Text>
                    <Text type="secondary" style={{ marginLeft: '8px' }}>
                      ({apiUtils.formatFileSize(uploadedFile.size)})
                    </Text>
                  </div>
                )}
              </div>
            </Col>
          
          <Col xs={24} lg={8}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', height: '100%' }}>
              <Button 
                type="primary" 
                icon={<UploadOutlined />}
                onClick={handleUpload}
                disabled={!uploadedFile}
                size="large"
                block
              >
                파일 업로드 및 로드
              </Button>
              
              <Button 
                icon={<FileTextOutlined />}
                onClick={loadDefaultData}
                size="large"
                block
              >
                기본 데이터 사용
              </Button>
              
              <div style={{ padding: '16px', background: '#f9f9f9', borderRadius: '6px', fontSize: '12px' }}>
                <Text strong>지원 형식:</Text>
                <br />• CSV 파일 (.csv)
                <br />• 최대 크기: 50MB
                <br />• 필수 컬럼: attrition
                <br />• Score 컬럼들 포함 권장
              </div>
            </div>
          </Col>
        </Row>

        {/* 업로드 진행률 */}
        {uploadProgress > 0 && (
          <div style={{ marginTop: 16 }}>
            <Progress percent={uploadProgress} status="active" />
          </div>
        )}
      </Card>
      )}

      {/* 데이터 유효성 검사 결과 */}
      {moduleType !== 'cognita' && validationResults && (
        <Card title="데이터 유효성 검사" className="card-shadow" style={{ marginBottom: 24 }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={8}>
              <Statistic
                title="전체 행 수"
                value={validationResults.dataQuality.totalRows}
                prefix={<FileTextOutlined />}
              />
            </Col>
            <Col xs={24} sm={8}>
              <Statistic
                title="유효한 행"
                value={validationResults.dataQuality.validRows}
                valueStyle={{ color: '#3f8600' }}
                prefix={<CheckCircleOutlined />}
              />
            </Col>
            <Col xs={24} sm={8}>
              <Statistic
                title="데이터 품질"
                value={`${((validationResults.dataQuality.validRows / validationResults.dataQuality.totalRows) * 100).toFixed(1)}%`}
                valueStyle={{ 
                  color: validationResults.dataQuality.validRows / validationResults.dataQuality.totalRows > 0.8 ? '#3f8600' : '#faad14'
                }}
              />
            </Col>
          </Row>

          <div style={{ marginTop: 16 }}>
            {!validationResults.hasRequiredColumns && (
              <Alert
                message="필수 컬럼 누락"
                description={`다음 컬럼이 필요합니다: ${validationResults.missingColumns.join(', ')}`}
                type="error"
                showIcon
                style={{ marginBottom: 8 }}
              />
            )}
            
            {validationResults.missingScoreColumns.length > 0 && (
              <Alert
                message="Score 컬럼 누락"
                description={`다음 Score 컬럼이 누락되었습니다: ${validationResults.missingScoreColumns.join(', ')}`}
                type="warning"
                showIcon
                style={{ marginBottom: 8 }}
              />
            )}
            
            {validationResults.dataQuality.invalidValues > 0 && (
              <Alert
                message="데이터 품질 경고"
                description={`${validationResults.dataQuality.invalidValues}개의 잘못된 값이 발견되었습니다.`}
                type="warning"
                showIcon
                style={{ marginBottom: 8 }}
              />
            )}
            
            {validationResults.hasRequiredColumns && validationResults.hasScoreColumns && validationResults.dataQuality.invalidValues === 0 && (
              <Alert
                message="데이터 검증 완료"
                description="모든 필수 컬럼이 존재하고 데이터 품질이 양호합니다."
                type="success"
                showIcon
              />
            )}
          </div>
        </Card>
      )}

      {/* 데이터 미리보기 */}
      {dataPreview && (
        <Card title="데이터 미리보기" className="card-shadow" style={{ marginBottom: 24 }}>
          <div style={{ marginBottom: 16 }}>
            <Text>
              <strong>컬럼 수:</strong> {dataPreview.headers.length} | 
              <strong> 미리보기 행 수:</strong> {dataPreview.data.length}
            </Text>
          </div>
          
          <Table
            columns={getTableColumns()}
            dataSource={dataPreview.data.map((row, index) => ({ ...row, key: index }))}
            pagination={false}
            scroll={{ x: true }}
            size="small"
            bordered
          />
          
          {dataPreview.data.length >= 10 && (
            <div style={{ textAlign: 'center', marginTop: 8, color: '#666' }}>
              <Text type="secondary">처음 10행만 표시됩니다.</Text>
            </div>
          )}
        </Card>
      )}

      {/* 로드된 데이터 정보 */}
      {moduleType !== 'cognita' && fileData && (
        <Card title="로드된 데이터 정보" className="card-shadow">
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={8}>
              <Statistic
                title="전체 행 수"
                value={fileData.statistics?.total_rows?.toLocaleString() || 0}
                prefix={<FileTextOutlined />}
              />
            </Col>
            <Col xs={24} sm={8}>
              <Statistic
                title="전체 컬럼 수"
                value={fileData.statistics?.total_columns || 0}
                prefix={<CheckCircleOutlined />}
              />
            </Col>
            <Col xs={24} sm={8}>
              <Statistic
                title="Score 컬럼 수"
                value={fileData.statistics?.score_columns?.length || 0}
                prefix={<ExclamationCircleOutlined />}
              />
            </Col>
          </Row>

          <div style={{ marginTop: 16 }}>
            <Title level={5}>Attrition 분포</Title>
            {fileData.statistics?.attrition_distribution && (
              <Row gutter={16}>
                {Object.entries(fileData.statistics.attrition_distribution).map(([key, value]) => (
                  <Col key={key} span={12}>
                    <div style={{ 
                      padding: '12px', 
                      background: key === 'Yes' ? '#fff2f0' : '#f6ffed',
                      border: `1px solid ${key === 'Yes' ? '#ffccc7' : '#b7eb8f'}`,
                      borderRadius: '6px',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '18px', fontWeight: 'bold' }}>
                        {value.toLocaleString()}
                      </div>
                      <div style={{ color: '#666' }}>
                        {key === 'Yes' ? 'Attrition' : 'No Attrition'}
                      </div>
                    </div>
                  </Col>
                ))}
              </Row>
            )}
          </div>

          <div style={{ marginTop: 16 }}>
            <Title level={5}>Score 컬럼</Title>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {fileData.statistics?.score_columns?.map(col => (
                <span key={col} style={{
                  padding: '4px 8px',
                  background: '#f0f8ff',
                  border: '1px solid #d9d9d9',
                  borderRadius: '4px',
                  fontSize: '12px'
                }}>
                  {col}
                </span>
              ))}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default FileUpload;
