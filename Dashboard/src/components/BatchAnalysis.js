import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Upload,
  message,
  Progress,
  Typography,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Alert,
  Space,
  Input,
  Slider,
  Modal,
  Spin,
  Descriptions,
  Divider,
  Badge,
  List
} from 'antd';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import {
  UploadOutlined,
  FileTextOutlined,
  ApiOutlined,
  BarChartOutlined,
  RocketOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  DownloadOutlined,
  TeamOutlined,
  HistoryOutlined,
  FilePdfOutlined,
  DashboardOutlined
} from '@ant-design/icons';
import { predictionService } from '../services/predictionService';

const { Title, Text, Paragraph } = Typography;
const { Dragger } = Upload;

const BatchAnalysis = ({
  loading,
  setLoading,
  serverStatus,
  onNavigate, // 네비게이션 콜백 함수 추가
  globalBatchResults, // 전역 배치 결과
  updateBatchResults // 배치 결과 업데이트 함수
}) => {
  const [analysisResults, setAnalysisResults] = useState(globalBatchResults);
  const [analysisProgress, setAnalysisProgress] = useState({
    structura: 0,
    cognita: 0,
    chronos: 0,
    sentio: 0,
    agora: 0,
    overall: 0
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [employeeData, setEmployeeData] = useState([]);
  
  // 각 에이전트별 데이터 상태
  const [agentFiles, setAgentFiles] = useState({
    structura: null,
    chronos: null,
    sentio: null,
    agora: null
  });
  
  // Neo4j 연결 설정
  const [neo4jConfig, setNeo4jConfig] = useState({
    uri: 'bolt://44.212.67.74:7687',
    username: 'neo4j',
    password: 'legs-augmentations-cradle'
  });
  
  // 사후 분석에서 저장된 최종 설정 상태
  const [finalRiskSettings, setFinalRiskSettings] = useState(null);
  const [neo4jConnected, setNeo4jConnected] = useState(false);
  const [neo4jTesting, setNeo4jTesting] = useState(false);
  
  // 개별 직원 레포트 관련 상태
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [employeeReportVisible, setEmployeeReportVisible] = useState(false);
  const [employeeReport, setEmployeeReport] = useState(null);
  const [reportGenerating, setReportGenerating] = useState(false);
  

  // 캐시 관련 상태
  const [cachedResults, setCachedResults] = useState([]);
  const [selectedCacheId, setSelectedCacheId] = useState(null);
  const [showCacheOptions, setShowCacheOptions] = useState(false);
  const [cacheModalVisible, setCacheModalVisible] = useState(false);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  
  // 결과 내보내기 관련 상태
  const [isExporting, setIsExporting] = useState(false);

  // 컴포넌트 로드 시 캐시 확인
  useEffect(() => {
    loadCachedResults();
  }, []);

  // 사후 분석 최종 설정 로드
  useEffect(() => {
    const savedSettings = localStorage.getItem('finalRiskSettings');
    if (savedSettings) {
      try {
        const settings = JSON.parse(savedSettings);
        setFinalRiskSettings(settings);
        console.log('📋 사후 분석 최종 설정 로드됨:', settings);
      } catch (error) {
        console.error('최종 설정 파싱 오류:', error);
      }
    }
  }, []);

  // 캐시된 결과 로드
  const loadCachedResults = async () => {
    try {
      const history = await predictionService.getPredictionHistoryAsync();
      setCachedResults(history);
      
      if (history.length > 0) {
        // 가장 최근 결과가 있으면 스마트 로딩 옵션 표시
        setShowCacheOptions(true);
        
        // 전역 상태에 결과가 없으면 자동으로 최신 캐시 로드
        if (!globalBatchResults && history.length > 0) {
          const latestCache = history[0];
          message.info(`최근 분석 결과를 발견했습니다 (${new Date(latestCache.timestamp).toLocaleString('ko-KR')})`);
        }
      }
    } catch (error) {
      console.error('캐시 로드 실패:', error);
    }
  };

  // 캐시된 결과 사용
  const loadCachedResult = (cacheId) => {
    const cachedResult = cachedResults.find(cache => cache.id === cacheId);
    if (cachedResult && cachedResult.rawData) {
      // 캐시된 원본 데이터를 현재 결과로 설정
      setAnalysisResults(cachedResult.rawData);
      updateBatchResults(cachedResult.rawData);
      
      message.success(`캐시된 분석 결과를 불러왔습니다 (${new Date(cachedResult.timestamp).toLocaleString('ko-KR')})`);
      setShowCacheOptions(false);
      setCacheModalVisible(false);
    } else {
      message.error('캐시된 데이터를 불러올 수 없습니다.');
    }
  };

  // 최신 캐시 자동 로드
  const loadLatestCache = () => {
    if (cachedResults.length > 0) {
      loadCachedResult(cachedResults[0].id);
    }
  };

  // 새로 분석 시작
  const startNewAnalysis = () => {
    setShowCacheOptions(false);
    message.info('새로운 분석을 시작합니다. 파일을 업로드해주세요.');
  };

  // 최적화된 설정을 적용한 분석 결과 해석
  const generateAnalysisInsights = (results) => {
    // analysisResults.results 배열을 사용하여 분석
    const employeeResults = results.results || results;
    const totalEmployees = employeeResults.length;
    
    // 사후 분석의 최적화된 설정 로드
    const finalSettings = finalRiskSettings || {};
    const optimizedThresholds = finalSettings.risk_thresholds || {};
    const optimizedWeights = finalSettings.integration_config || {};
    
    console.log('📊 배치 분석에 적용된 최적화 설정:', {
      thresholds: optimizedThresholds,
      weights: optimizedWeights,
      prediction_mode: finalSettings.attrition_prediction_mode
    });
    
    // 각 직원의 위험도 계산 및 부서 정보 추출
    const processedEmployees = employeeResults.map(emp => {
      // 최적화된 가중치가 있으면 적용, 없으면 기본 통합 점수 사용
      let riskScore = emp.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
      
      // 최적화된 가중치로 재계산 (가능한 경우)
      if (optimizedWeights && Object.keys(optimizedWeights).length > 0) {
        const structuraScore = emp.analysis_result?.structura_result?.prediction?.attrition_probability || 0;
        const cognitaScore = emp.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score || 0;
        const chronosScore = emp.analysis_result?.chronos_result?.prediction?.risk_score || 0;
        const sentioScore = emp.analysis_result?.sentio_result?.sentiment_analysis?.risk_score || 0;
        const agoraScore = emp.analysis_result?.agora_result?.market_analysis?.risk_score || 0;
        
        riskScore = (
          (structuraScore * (optimizedWeights.structura_weight || 0.3)) +
          (cognitaScore * (optimizedWeights.cognita_weight || 0.2)) +
          (chronosScore * (optimizedWeights.chronos_weight || 0.2)) +
          (sentioScore * (optimizedWeights.sentio_weight || 0.15)) +
          (agoraScore * (optimizedWeights.agora_weight || 0.15))
        );
      }
      
      // 최적화된 임계값으로 위험도 분류
      const riskLevel = riskScore ? calculateRiskLevel(riskScore) : 'UNKNOWN';
      
      // 부서 정보 추출 (다양한 소스에서 시도, 부서 분류 문제 해결)
      let department = 'Unclassified';
      
      // 직원 데이터에서 부서 정보 찾기 (우선순위 순)
      const sources = [
        emp.analysis_result?.employee_data?.Department,
        emp.department,
        emp.analysis_result?.structura_result?.employee_data?.Department,
        emp.employee_data?.Department,
        emp.Department  // 직접 부서 필드
      ];
      
      // 첫 번째로 유효한 부서 정보 사용
      for (const source of sources) {
        if (source && typeof source === 'string' && source.trim() && source !== '미분류') {
          department = source.trim();
          break;
        }
      }
      
      // 부서명 정규화 (일반적인 부서명으로 매핑)
      const deptMapping = {
        'HR': 'Human Resources',
        'IT': 'Information Technology', 
        'R&D': 'Research and Development',
        'Sales': 'Sales',
        'Marketing': 'Marketing',
        'Finance': 'Finance',
        'Operations': 'Operations',
        'Manufacturing': 'Manufacturing'
      };
      
      // 매핑된 부서명이 있으면 사용
      if (deptMapping[department]) {
        department = deptMapping[department];
      }
      
      // 디버깅: 부서 정보 추출 과정 로그
      if (emp.employee_number && (emp.employee_number.endsWith('001') || emp.employee_number.endsWith('002'))) {
        console.log(`🔍 직원 ${emp.employee_number} 부서 정보:`, {
          'analysis_result.employee_data.Department': emp.analysis_result?.employee_data?.Department,
          'department': emp.department,
          'structura.employee_data.Department': emp.analysis_result?.structura?.employee_data?.Department,
          'employee_data.Department': emp.employee_data?.Department,
          'final_department': department
        });
      }
      
      return {
        ...emp,
        risk_level: riskLevel.toLowerCase(),
        risk_score: riskScore,
        department: department
      };
    });

    // 최적화된 위험도 분류 통계 (사후 분석 기준 적용)
    const highRisk = processedEmployees.filter(emp => emp.risk_level === 'high').length;
    const mediumRisk = processedEmployees.filter(emp => emp.risk_level === 'medium').length;
    const lowRisk = processedEmployees.filter(emp => emp.risk_level === 'low').length;
    
    // 최적화된 퇴사 예측 (사후 분석 설정 적용)
    const predictionMode = finalSettings.attrition_prediction_mode || 'high_risk_only';
    let predictedAttrition = 0;
    
    if (predictionMode === 'high_risk_only') {
      predictedAttrition = highRisk; // 고위험군만 퇴사 예측
    } else if (predictionMode === 'medium_high_risk') {
      predictedAttrition = highRisk + mediumRisk; // 주의군 + 고위험군 퇴사 예측
    }
    
    console.log('📊 최적화된 위험도 분류 결과:', {
      안전군: lowRisk,
      주의군: mediumRisk, 
      고위험군: highRisk,
      퇴사예측모드: predictionMode,
      예측퇴사자수: predictedAttrition,
      임계값: optimizedThresholds
    });

    // 부서별 통계
    const departmentStats = {};
    processedEmployees.forEach(emp => {
      const dept = emp.department || '미분류';
      if (!departmentStats[dept]) {
        departmentStats[dept] = { total: 0, high: 0, medium: 0, low: 0 };
      }
      departmentStats[dept].total++;
      departmentStats[dept][emp.risk_level]++;
    });

    // 주요 위험 요인 분석 (객체 직렬화 오류 및 빈 데이터 문제 해결)
    const riskFactors = {};
    processedEmployees.filter(emp => emp.risk_level === 'high').forEach(emp => {
      // 위험 요인을 분석 결과에서 추출
      const factors = [];
      
      // Structura 위험 요인 (안전한 추출)
      if (emp.analysis_result?.structura_result?.explanation?.top_risk_factors) {
        emp.analysis_result.structura_result.explanation.top_risk_factors.forEach(factor => {
          if (factor && typeof factor === 'object' && factor.feature) {
            factors.push(`Structura: ${factor.feature}`);
          }
        });
      }
      
      // Cognita 위험 요인 (안전한 추출)
      if (emp.analysis_result?.cognita_result?.risk_analysis?.risk_factors) {
        const cognitaFactors = emp.analysis_result.cognita_result.risk_analysis.risk_factors;
        if (Array.isArray(cognitaFactors)) {
          cognitaFactors.forEach(factor => {
            if (factor && typeof factor === 'string') {
              factors.push(`Cognita: ${factor}`);
            }
          });
        }
      }
      
      // 통합 분석의 위험 요인 (안전한 추출)
      if (emp.analysis_result?.combined_analysis?.risk_factors) {
        const combinedFactors = emp.analysis_result.combined_analysis.risk_factors;
        if (Array.isArray(combinedFactors)) {
          combinedFactors.forEach(factor => {
            if (factor && typeof factor === 'string') {
              factors.push(`Combined: ${factor}`);
            }
          });
        }
      }
      
      // 기본 위험 요인 (점수 기반)
      if (emp.risk_score > 0.8) {
        factors.push('Very High Risk Score');
      } else if (emp.risk_score > 0.7) {
        factors.push('High Risk Score');
      }
      
      // 부서별 위험 요인
      if (emp.department && emp.department !== 'Unclassified') {
        factors.push(`Department: ${emp.department}`);
      }
      
      // 위험 요인이 없으면 기본 요인 추가
      if (factors.length === 0) {
        factors.push('General Risk Factors');
      }
      
      factors.forEach(factor => {
        if (factor && typeof factor === 'string' && factor.trim()) {
          const cleanFactor = factor.trim();
          riskFactors[cleanFactor] = (riskFactors[cleanFactor] || 0) + 1;
        }
      });
    });

    // 디버깅: 부서별 통계 출력
    console.log('🏢 부서별 통계 분석:', departmentStats);
    console.log('👥 처리된 직원 수:', processedEmployees.length);
    console.log('📊 부서 분포:', Object.keys(departmentStats));

    const insights = {
      summary: {
        totalEmployees,
        highRisk,
        mediumRisk,
        lowRisk,
        highRiskPercentage: ((highRisk / totalEmployees) * 100).toFixed(1),
        mediumRiskPercentage: ((mediumRisk / totalEmployees) * 100).toFixed(1),
        lowRiskPercentage: ((lowRisk / totalEmployees) * 100).toFixed(1)
      },
      departmentAnalysis: departmentStats,
      topRiskFactors: Object.entries(riskFactors)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .map(([factor, count]) => ({ factor, count })),
      recommendations: generateRecommendations(highRisk, mediumRisk, totalEmployees, departmentStats),
      riskTrends: analyzeTrends(processedEmployees),
      criticalAlerts: generateCriticalAlerts(processedEmployees, departmentStats)
    };

    return insights;
  };

  // 추천 사항 생성 (권장사항 생성 로직 개선)
  const generateRecommendations = (highRisk, mediumRisk, total, deptStats) => {
    const recommendations = [];
    
    // 전체 고위험 비율 기반 권장사항
    if (total > 0) {
      const highRiskPercentage = (highRisk / total) * 100;
      
      if (highRiskPercentage > 20) {
        recommendations.push({
          priority: 'high',
          title: 'Immediate Action Required',
          description: `${highRiskPercentage.toFixed(1)}% of employees are in high-risk category. Implement immediate individual interviews and support programs.`
        });
      } else if (highRiskPercentage > 10) {
        recommendations.push({
          priority: 'medium',
          title: 'Enhanced Monitoring Required',
          description: `${highRiskPercentage.toFixed(1)}% of employees are in high-risk category. Consider enhanced monitoring and preventive measures.`
        });
      }
      
      // 중위험 비율 기반 권장사항
      const mediumRiskPercentage = (mediumRisk / total) * 100;
      if (mediumRiskPercentage > 30) {
        recommendations.push({
          priority: 'medium',
          title: 'Preventive Measures Recommended',
          description: `${mediumRiskPercentage.toFixed(1)}% of employees are in medium-risk category. Introduce preventive mentoring programs.`
        });
      }
    }

    // 부서별 권장사항 (안전한 처리)
    if (deptStats && typeof deptStats === 'object') {
      Object.entries(deptStats).forEach(([dept, stats]) => {
        if (stats && typeof stats === 'object' && stats.total > 0) {
          const deptHighRiskPercentage = (stats.high / stats.total) * 100;
          
          if (deptHighRiskPercentage > 30) {
            recommendations.push({
              priority: 'high',
              title: `${dept} Department Focus Required`,
              description: `${dept} department has ${deptHighRiskPercentage.toFixed(1)}% high-risk employees. Department-specific intervention needed.`
            });
          } else if (deptHighRiskPercentage > 20) {
            recommendations.push({
              priority: 'medium',
              title: `${dept} Department Monitoring`,
              description: `${dept} department shows elevated risk levels (${deptHighRiskPercentage.toFixed(1)}%). Enhanced monitoring recommended.`
            });
          }
        }
      });
    }
    
    // 기본 권장사항 (권장사항이 없을 경우)
    if (recommendations.length === 0) {
      recommendations.push({
        priority: 'low',
        title: 'Continue Regular Monitoring',
        description: 'Current risk levels are within acceptable ranges. Continue regular monitoring and maintain current retention strategies.'
      });
    }

    return recommendations;
  };

  // 트렌드 분석
  const analyzeTrends = (results) => {
    // 간단한 트렌드 분석 (실제로는 시계열 데이터가 필요)
    return {
      overallTrend: 'stable',
      departmentTrends: {},
      seasonalPatterns: []
    };
  };

  // 중요 알림 생성
  const generateCriticalAlerts = (processedEmployees, deptStats) => {
    const alerts = [];
    
    const highRiskEmployees = processedEmployees.filter(emp => emp.risk_level === 'high');
    if (highRiskEmployees.length > 0) {
      alerts.push({
        type: 'critical',
        message: `${highRiskEmployees.length}명의 직원이 즉시 관심이 필요한 고위험 상태입니다.`
      });
    }

    // 부서별 알림
    Object.entries(deptStats).forEach(([dept, stats]) => {
      if (stats.high > 5) {
        alerts.push({
          type: 'warning',
          message: `${dept} 부서에서 ${stats.high}명의 고위험 직원이 발견되었습니다.`
        });
      }
    });

    return alerts;
  };

  // PDF 보고서 생성
  const generatePDFReport = async () => {
    if (!analysisResults || !analysisResults.results || analysisResults.results.length === 0) {
      message.error('분석 결과가 없습니다. 먼저 배치 분석을 실행해주세요.');
      return;
    }

    setIsGeneratingPDF(true);
    
    try {
      // LLM 분석 수행
      const insights = generateAnalysisInsights(analysisResults);
      
      // PDF 생성
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      let yPosition = 20;

      // 한글 지원을 위한 폰트 설정
      pdf.setFont('helvetica');

      // 제목
      pdf.setFontSize(20);
      pdf.setTextColor(0, 0, 0);
      const title = 'Batch Analysis Report';
      pdf.text(title, pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 15;

      // 생성 날짜 (한국어 인코딩 문제 해결)
      pdf.setFontSize(12);
      const now = new Date();
      const timestamp = `${now.getFullYear()}. ${(now.getMonth() + 1).toString().padStart(2, '0')}. ${now.getDate().toString().padStart(2, '0')} ${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
      pdf.text(`Generated: ${timestamp}`, pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 20;

      // 요약 섹션
      pdf.setFontSize(16);
      pdf.setTextColor(0, 0, 0);
      pdf.text('Executive Summary', 20, yPosition);
      yPosition += 10;

      pdf.setFontSize(11);
      const summaryLines = [
        `Total Employees Analyzed: ${insights.summary.totalEmployees}`,
        `High Risk: ${insights.summary.highRisk} (${insights.summary.highRiskPercentage}%)`,
        `Medium Risk: ${insights.summary.mediumRisk} (${insights.summary.mediumRiskPercentage}%)`,
        `Low Risk: ${insights.summary.lowRisk} (${insights.summary.lowRiskPercentage}%)`
      ];

      summaryLines.forEach(line => {
        pdf.text(line, 20, yPosition);
        yPosition += 7;
      });

      yPosition += 10;

      // 부서별 분석
      pdf.setFontSize(16);
      pdf.text('Department Analysis', 20, yPosition);
      yPosition += 10;

      pdf.setFontSize(11);
      Object.entries(insights.departmentAnalysis).forEach(([dept, stats]) => {
        if (yPosition > pageHeight - 30) {
          pdf.addPage();
          yPosition = 20;
        }
        
        // 부서명을 영어로 변환하거나 안전하게 처리
        const safeDeptName = dept === '미분류' ? 'Unclassified' : 
                           dept.replace(/[^\x00-\x7F]/g, '') || 'Department';
        
        pdf.text(`${safeDeptName}: Total ${stats.total}, High Risk ${stats.high}, Medium Risk ${stats.medium}, Low Risk ${stats.low}`, 20, yPosition);
        yPosition += 7;
      });

      yPosition += 10;

      // 주요 위험 요인
      if (insights.topRiskFactors && insights.topRiskFactors.length > 0) {
        pdf.setFontSize(16);
        pdf.text('Top Risk Factors', 20, yPosition);
        yPosition += 10;

        pdf.setFontSize(11);
        insights.topRiskFactors.forEach(({ factor, count }) => {
          if (yPosition > pageHeight - 30) {
            pdf.addPage();
            yPosition = 20;
          }
          
          // 위험 요인을 안전하게 처리 (객체 직렬화 오류 해결)
          let safeFactor = 'Risk Factor';
          if (typeof factor === 'string' && factor.trim()) {
            safeFactor = factor.replace(/[^\x00-\x7F]/g, '').trim() || 'Risk Factor';
          } else if (typeof factor === 'object' && factor !== null) {
            safeFactor = JSON.stringify(factor).replace(/[^\x00-\x7F]/g, '') || 'Risk Factor';
          }
          
          const safeCount = typeof count === 'number' ? count : 0;
          pdf.text(`${safeFactor}: ${safeCount} employees`, 20, yPosition);
          yPosition += 7;
        });

        yPosition += 10;
      }

      // 권장사항
      if (insights.recommendations && insights.recommendations.length > 0) {
        pdf.setFontSize(16);
        pdf.text('Recommendations', 20, yPosition);
        yPosition += 10;

        pdf.setFontSize(11);
        insights.recommendations.forEach((rec, index) => {
          if (yPosition > pageHeight - 40) {
            pdf.addPage();
            yPosition = 20;
          }
          
          // 우선순위에 따른 색상 설정
          const priority = rec.priority || 'medium';
          pdf.setTextColor(priority === 'high' ? 255 : 0, priority === 'medium' ? 165 : 0, 0);
          
          // 제목을 안전하게 처리 (빈 데이터 문제 해결)
          let safeTitle = 'Recommendation';
          if (typeof rec.title === 'string' && rec.title.trim()) {
            safeTitle = rec.title.replace(/[^\x00-\x7F]/g, '').trim() || 'Recommendation';
          }
          
          pdf.text(`${index + 1}. ${safeTitle}`, 20, yPosition);
          yPosition += 7;
          
          pdf.setTextColor(0, 0, 0);
          
          // 설명을 안전하게 처리 (빈 데이터 문제 해결)
          let safeDescription = 'Please review this recommendation.';
          if (typeof rec.description === 'string' && rec.description.trim()) {
            safeDescription = rec.description.replace(/[^\x00-\x7F]/g, '').trim() || 'Please review this recommendation.';
          }
          
          const descLines = pdf.splitTextToSize(safeDescription, pageWidth - 40);
          descLines.forEach(line => {
            if (line.trim()) { // 빈 줄 제거
              pdf.text(line, 25, yPosition);
              yPosition += 6;
            }
          });
          yPosition += 5;
        });
      }

      // 중요 알림
      if (insights.criticalAlerts && insights.criticalAlerts.length > 0) {
        if (yPosition > pageHeight - 50) {
          pdf.addPage();
          yPosition = 20;
        }

        pdf.setFontSize(16);
        pdf.setTextColor(255, 0, 0);
        pdf.text('Critical Alerts', 20, yPosition);
        yPosition += 10;

        pdf.setFontSize(11);
        insights.criticalAlerts.forEach(alert => {
          const alertType = alert.type || 'warning';
          pdf.setTextColor(alertType === 'critical' ? 255 : 255, alertType === 'critical' ? 0 : 165, 0);
          
          // 알림 메시지를 안전하게 처리 (빈 데이터 문제 해결)
          let safeMessage = 'Critical alert requires attention.';
          if (typeof alert.message === 'string' && alert.message.trim()) {
            safeMessage = alert.message.replace(/[^\x00-\x7F]/g, '').trim() || 'Critical alert requires attention.';
          }
          
          const alertLines = pdf.splitTextToSize(safeMessage, pageWidth - 40);
          alertLines.forEach(line => {
            if (line.trim()) { // 빈 줄 제거
              pdf.text(line, 20, yPosition);
              yPosition += 6;
            }
          });
          yPosition += 5;
        });
      }

      // 파일명 생성
      const fileName = `Report_BatchAnalysis_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.pdf`;
      
      // PDF 저장
      pdf.save(fileName);
      
      message.success(`PDF 보고서가 생성되었습니다: ${fileName}`);
      
    } catch (error) {
      console.error('PDF 생성 오류:', error);
      message.error('PDF 보고서 생성 중 오류가 발생했습니다.');
    } finally {
      setIsGeneratingPDF(false);
    }
  };
  
  
  // Integration 설정 (사후 분석에서 최적화된 값 자동 로드)
  const [integrationConfig, setIntegrationConfig] = useState({
    // 에이전트별 가중치 (기본값 - 사후 분석 후 최적화 예정)
    structura_weight: 0.25,  // 정형 데이터 분석
    cognita_weight: 0.20,    // 관계형 데이터 분석
    chronos_weight: 0.25,    // 시계열 데이터 분석
    sentio_weight: 0.15,     // 감정 분석
    agora_weight: 0.15,      // 시장 분석
    
    // 위험도 분류 임계값 (기본값)
    high_risk_threshold: 0.7,
    medium_risk_threshold: 0.4,
    
    // 개별 에이전트 임계값 (기본값 - 사후 분석을 통해 최적화 예정)
    structura_threshold: 0.7,
    cognita_threshold: 0.5,
    chronos_threshold: 0.5,
    sentio_threshold: 0.5,
    agora_threshold: 0.5,
    
    // 모델 사용 설정
    use_trained_models: false
  });

  // 컴포넌트 로드 시 저장된 모델 확인 및 설정 자동 로드
  useEffect(() => {
    loadTrainedModels();
  }, []);

  // 저장된 모델 로드 함수
  const loadTrainedModels = () => {
    try {
      const savedModels = localStorage.getItem('trainedModels');
      if (savedModels) {
        const modelData = JSON.parse(savedModels);
        console.log('💾 저장된 모델 발견:', modelData);
        
        // 최적화된 설정이 있으면 자동 적용
        if (modelData.optimization_results) {
          const optimized = modelData.optimization_results;
          
          setIntegrationConfig(prev => ({
            ...prev,
            // 최적화된 가중치 적용
            structura_weight: optimized.weights?.structura_weight || prev.structura_weight,
            cognita_weight: optimized.weights?.cognita_weight || prev.cognita_weight,
            chronos_weight: optimized.weights?.chronos_weight || prev.chronos_weight,
            sentio_weight: optimized.weights?.sentio_weight || prev.sentio_weight,
            agora_weight: optimized.weights?.agora_weight || prev.agora_weight,
            
            // 최적화된 임계값 적용
            high_risk_threshold: optimized.thresholds?.high_risk_threshold || prev.high_risk_threshold,
            medium_risk_threshold: optimized.thresholds?.medium_risk_threshold || prev.medium_risk_threshold,
            structura_threshold: optimized.thresholds?.structura_threshold || prev.structura_threshold,
            cognita_threshold: optimized.thresholds?.cognita_threshold || prev.cognita_threshold,
            chronos_threshold: optimized.thresholds?.chronos_threshold || prev.chronos_threshold,
            sentio_threshold: optimized.thresholds?.sentio_threshold || prev.sentio_threshold,
            agora_threshold: optimized.thresholds?.agora_threshold || prev.agora_threshold,
            
            use_trained_models: true
          }));
          
          message.success(
            `사후 분석에서 최적화된 설정을 자동으로 적용했습니다! ` +
            `(학습일: ${new Date(modelData.training_metadata?.training_date).toLocaleDateString('ko-KR')})`
          );
        }
      }
    } catch (error) {
      console.error('저장된 모델 로드 실패:', error);
    }
  };


  // 에이전트별 파일 업로드 처리
  const handleAgentFileUpload = async (file, agentType) => {
    const isCSV = file.type === 'text/csv' || file.name.endsWith('.csv');
    if (!isCSV) {
      message.error('CSV 파일만 업로드 가능합니다.');
      return false;
    }

    // 파일 크기 제한 (Chronos는 500MB, 나머지는 10MB)
    const maxSize = agentType === 'chronos' ? 500 : 10;
    const isLtMaxSize = file.size / 1024 / 1024 < maxSize;
    if (!isLtMaxSize) {
      message.error(`파일 크기는 ${maxSize}MB 이하여야 합니다.`);
      return false;
    }

    try {
      setLoading(true);
      
      // CSV 파일 읽기 및 검증
      const text = await file.text();
      const lines = text.split('\n').filter(line => line.trim());
      
      if (lines.length < 2) {
        message.error('유효한 CSV 데이터가 없습니다.');
        return false;
      }

      // 에이전트별 필수 컬럼 검증
      const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
      const requiredColumns = getRequiredColumns(agentType);
      const missingColumns = requiredColumns.filter(col => !headers.includes(col));
      
      if (missingColumns.length > 0) {
        message.error(`필수 컬럼이 누락되었습니다: ${missingColumns.join(', ')}`);
        return false;
      }

      // 1. 먼저 파일을 Supervisor에 업로드
      const formData = new FormData();
      formData.append('file', file);
      formData.append('agent_type', agentType);
      formData.append('analysis_type', 'batch'); // 배치 분석용
      
      const uploadResponse = await fetch('http://localhost:5006/upload_file', {
        method: 'POST',
        body: formData
      });
      
      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        throw new Error(errorData.error || '파일 업로드 실패');
      }
      
      const uploadResult = await uploadResponse.json();
      console.log(`${agentType} 파일 업로드 성공:`, uploadResult);

      // 2. 파일 저장
      setAgentFiles(prev => ({
        ...prev,
        [agentType]: file
      }));

      // 3. Structura 파일인 경우 직원 데이터도 파싱
      if (agentType === 'structura') {
        const employees = parseEmployeeData(lines, headers);
        setEmployeeData(employees);
      }

      message.success(`${agentType} 데이터를 업로드하고 로드했습니다.`);
      return false; // Ant Design Upload 컴포넌트의 자동 업로드 방지
      
    } catch (error) {
      console.error(`${agentType} 파일 업로드 실패:`, error);
      message.error(`${agentType} 파일 업로드 실패: ${error.message}`);
      return false;
    } finally {
      setLoading(false);
    }
  };

  // 에이전트별 필수 컬럼 정의
  const getRequiredColumns = (agentType) => {
    const columnMap = {
      structura: ['EmployeeNumber', 'Age', 'JobRole', 'Department'],
      chronos: ['employee_id', 'date', 'work_focused_ratio', 'meeting_collaboration_ratio'],
      sentio: ['EmployeeNumber', 'SELF_REVIEW_text', 'PEER_FEEDBACK_text', 'WEEKLY_SURVEY_text'],
      agora: ['EmployeeNumber', 'Age', 'JobRole', 'Department'] // Structura와 동일한 데이터 사용
    };
    return columnMap[agentType] || [];
  };

  // 직원 데이터 파싱
  const parseEmployeeData = (lines, headers) => {
    const employees = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
      const employee = {};
      
      headers.forEach((header, index) => {
        let value = values[index] || '';
        
        // 숫자 컬럼 처리
        if (['Age', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance'].includes(header)) {
          value = parseFloat(value) || 0;
        }
        
        employee[header] = value;
      });
      
      if (employee.EmployeeNumber) {
        employees.push(employee);
      }
    }
    return employees;
  };

  // Neo4j 연결 테스트
  const testNeo4jConnection = async () => {
    if (!neo4jConfig.uri || !neo4jConfig.username || !neo4jConfig.password) {
      message.error('Neo4j 연결 정보를 모두 입력해주세요.');
      return;
    }

    setNeo4jTesting(true);
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
      
      // 응답 형식 통일 (success 필드 확인)
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
      setNeo4jTesting(false);
    }
  };

  // 에이전트 디버깅 정보 조회
  const debugAgents = async () => {
    try {
      console.log('🔍 에이전트 디버깅 정보 조회 시작');
      
      const response = await fetch('/api/agents/debug');
      const debugInfo = await response.json();
      
      console.log('🔍 에이전트 디버깅 정보:', debugInfo);
      
      // Console에 상세 정보 출력
      console.log('📊 에이전트별 상태:');
      Object.entries(debugInfo.agents).forEach(([agentName, info]) => {
        console.log(`  ${agentName}:`);
        console.log(`    - Import 가능: ${info.import_available}`);
        console.log(`    - 초기화됨: ${info.initialized}`);
        console.log(`    - 객체 존재: ${info.agent_object}`);
        if (info.error_message) {
          console.log(`    - 오류: ${info.error_message}`);
        }
      });
      
      message.success('에이전트 디버깅 정보가 Console에 출력되었습니다. F12를 눌러 확인하세요.');
      
    } catch (error) {
      console.error('❌ 에이전트 디버깅 정보 조회 실패:', error);
      message.error(`디버깅 정보 조회 실패: ${error.message}`);
    }
  };

  // 통합 배치 분석 실행 - 데이터 검증 + 순차적 워크플로우 + Integration
  const runBatchAnalysis = async () => {
    // 1. 필수 데이터 검증
    if (!agentFiles.structura) {
      message.error('Structura 데이터(HR 기본 데이터)를 업로드해주세요.');
      return;
    }

    // 2. 가중치 합계 검증
    const weightSum = integrationConfig.structura_weight + 
                     integrationConfig.cognita_weight + 
                     integrationConfig.chronos_weight + 
                     integrationConfig.sentio_weight + 
                     integrationConfig.agora_weight;
    
    if (Math.abs(weightSum - 1.0) > 0.01) {
      message.error(`가중치 합계가 1.0이 되어야 합니다. 현재: ${weightSum.toFixed(2)}`);
      return;
    }

    // 진행률 폴링을 위한 변수들을 함수 스코프에서 선언
    let progressInterval = null;
    let cleanup = null;

    try {
      setIsAnalyzing(true);
      setAnalysisProgress({
        structura: 0,
        cognita: 0,
        chronos: 0,
        sentio: 0,
        agora: 0,
        overall: 0
      });
      
      // 3. 에이전트별 활성화 여부 결정
      const agentConfig = {
        use_structura: !!agentFiles.structura,
        use_cognita: neo4jConnected,
        use_chronos: !!agentFiles.chronos,
        use_sentio: !!agentFiles.sentio,
        use_agora: !!agentFiles.agora || !!agentFiles.structura // Agora는 별도 파일 또는 Structura 데이터 사용
      };

      // Console 로그 추가
      console.log('🚀 배치 분석 시작');
      console.log('📊 직원 데이터:', employeeData.length, '명');
      console.log('🔧 에이전트 설정:', agentConfig);
      console.log('⚙️ 통합 설정:', integrationConfig);
      console.log('🔗 Neo4j 연결 상태:', neo4jConnected);

      // 4. 실제 진행률 폴링 시작
      progressInterval = setInterval(async () => {
        try {
          console.log('📊 진행률 조회 시도...');
          const progressResponse = await fetch('http://localhost:5006/batch_status');  // 배치 상태 확인
          console.log('📊 진행률 응답 상태:', progressResponse.status);
          
          if (progressResponse.ok) {
            const progressData = await progressResponse.json();
            console.log('📊 진행률 데이터:', progressData);
            
            if (progressData.success) {
              // 진행률을 0-100 범위로 정규화하고 소수점 둘째자리까지 표시
              const normalizeProgress = (value) => {
                if (typeof value === 'string' && value.includes('/')) {
                  const [current, total] = value.split('/').map(Number);
                  return total > 0 ? Math.min(100, (current / total) * 100) : 0;
                }
                return Math.min(100, Number(value) || 0);
              };

              setAnalysisProgress({
                structura: parseFloat(normalizeProgress(progressData.agent_progress?.structura).toFixed(2)),
                cognita: parseFloat(normalizeProgress(progressData.agent_progress?.cognita).toFixed(2)),
                chronos: parseFloat(normalizeProgress(progressData.agent_progress?.chronos).toFixed(2)),
                sentio: parseFloat(normalizeProgress(progressData.agent_progress?.sentio).toFixed(2)),
                agora: parseFloat(normalizeProgress(progressData.agent_progress?.agora).toFixed(2)),
                overall: parseFloat(normalizeProgress(progressData.overall_progress).toFixed(2))
              });
              
              // 분석 완료 시 폴링 중단
              if (progressData.status === 'completed') {
                clearInterval(progressInterval);
              }
            }
          }
        } catch (error) {
          console.error('진행률 조회 실패:', error);
        }
      }, 1000); // 1초마다 폴링
      
      // 컴포넌트 언마운트 시 폴링 정리
      cleanup = () => {
        if (progressInterval) {
          clearInterval(progressInterval);
        }
      };
      
      // 분석 완료 후 정리
      window.addEventListener('beforeunload', cleanup);

      // 5. 저장된 모델 정보 포함하여 배치 분석 API 호출
      let savedModelInfo = null;
      if (integrationConfig.use_trained_models) {
        try {
          const savedModels = localStorage.getItem('trainedModels');
          if (savedModels) {
            savedModelInfo = JSON.parse(savedModels);
            console.log('🧠 저장된 모델 사용:', savedModelInfo.training_metadata);
          }
        } catch (error) {
          console.error('저장된 모델 로드 실패:', error);
        }
      }

      const response = await fetch('http://localhost:5006/batch_analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          employees: employeeData,
          analysis_type: 'batch', // 배치 분석 타입 전달
          ...agentConfig,
          integration_config: integrationConfig,
          neo4j_config: neo4jConnected ? neo4jConfig : null,
          agent_files: {
            structura: agentFiles.structura?.name,
            chronos: agentFiles.chronos?.name,
            sentio: agentFiles.sentio?.name,
            agora: agentFiles.agora?.name
          },
          // 저장된 모델 정보 전달
          trained_models: savedModelInfo?.saved_models || null,
          use_trained_models: integrationConfig.use_trained_models
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const batchResult = await response.json();
      
      console.log('📥 배치 분석 응답 받음:', batchResult);
      
      if (batchResult.error) {
        console.error('❌ 배치 분석 오류:', batchResult.error);
        throw new Error(batchResult.error);
      }

      // 결과 분석 로깅
      if (batchResult.results) {
        const successCount = batchResult.results.filter(r => r.analysis_result && r.analysis_result.status === 'success').length;
        const failureCount = batchResult.results.filter(r => r.analysis_result && r.analysis_result.status === 'failed').length;
        console.log(`📈 분석 결과: 성공 ${successCount}명, 실패 ${failureCount}명`);
        
        // 실패한 케이스들 상세 로깅
        const failures = batchResult.results.filter(r => r.analysis_result && r.analysis_result.status === 'failed');
        if (failures.length > 0) {
          console.log('❌ 실패한 분석들:');
          failures.forEach((failure, index) => {
            console.log(`  ${index + 1}. 직원 ${failure.employee_number}:`, failure.analysis_result);
          });
        }
      }

      // 폴링 정리
      if (progressInterval) {
        clearInterval(progressInterval);
      }
      window.removeEventListener('beforeunload', cleanup);

      // 5. Integration 보고서 생성
      const reportResponse = await fetch('http://localhost:5006/api/workers/integration/generate_report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysis_results: batchResult.results,
          report_options: {
            include_recommendations: true,
            include_risk_analysis: true,
            integration_config: integrationConfig
          }
        })
      });

      if (!reportResponse.ok) {
        throw new Error(`Integration 보고서 생성 실패: ${reportResponse.status}`);
      }

      const reportResult = await reportResponse.json();
      
      if (reportResult.error) {
        throw new Error(reportResult.error);
      }

      // 6. 최종 결과 설정
      const finalResults = {
        ...batchResult,
        integration_report: reportResult.report,
        report_metadata: reportResult.metadata
      };
      
      setAnalysisResults(finalResults);
      
      // 전역 상태 업데이트 (다른 페이지에서 사용할 수 있도록)
      if (updateBatchResults) {
        updateBatchResults(finalResults);
      }

      // 예측 히스토리에 저장
      try {
        const predictionData = predictionService.convertBatchResultToPrediction(finalResults);
        if (predictionData) {
          predictionService.savePredictionResult(predictionData);
          message.success('분석 결과가 히스토리에 저장되었습니다.');
        }
      } catch (error) {
        console.error('예측 히스토리 저장 실패:', error);
        // 에러가 발생해도 메인 분석 프로세스는 계속 진행
      }

      // 부서별 결과 저장 (XAI 포함)
      try {
        console.log('💾 부서별 배치 분석 결과 저장 시작...');
        
        const saveResponse = await fetch('http://localhost:5007/api/batch-analysis/save-results', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            results: finalResults.results || [],
            applied_settings: finalRiskSettings || {},
            analysis_metadata: {
              total_employees: finalResults.total_employees,
              completed_employees: finalResults.completed_employees,
              analysis_timestamp: new Date().toISOString(),
              agents_used: Object.keys(agentConfig).filter(key => agentConfig[key] && key.startsWith('use_')),
              integration_config: integrationConfig
            }
          })
        });

        if (saveResponse.ok) {
          const saveResult = await saveResponse.json();
          console.log('✅ 부서별 결과 저장 완료:', saveResult);
          
          message.success(
            `부서별 분석 결과 저장 완료! ` +
            `${saveResult.statistics?.total_departments || 0}개 부서, ` +
            `${saveResult.statistics?.total_employees || 0}명 직원 (XAI PNG 시각화 포함)`
          );
        } else {
          const errorResult = await saveResponse.json();
          console.error('❌ 부서별 결과 저장 실패:', errorResult);
        }
      } catch (error) {
        console.error('부서별 결과 저장 중 오류:', error);
        // 에러가 발생해도 메인 분석 프로세스는 계속 진행
      }
      
      setAnalysisProgress(prev => ({ ...prev, overall: 100 }));

      message.success(`통합 배치 분석 완료! ${batchResult.completed_employees}명의 직원 분석 및 Integration 보고서 생성이 완료되었습니다.`);

    } catch (error) {
      console.error('❌ 통합 배치 분석 실패:', error);
      console.error('❌ 오류 스택:', error.stack);
      
      // 네트워크 오류인지 확인
      if (error.message.includes('fetch')) {
        console.error('🌐 네트워크 연결 문제가 발생했습니다. 백엔드 서버가 실행 중인지 확인하세요.');
        message.error('네트워크 연결 문제가 발생했습니다. 백엔드 서버 상태를 확인하세요.');
      } else {
        message.error(`통합 배치 분석 실패: ${error.message}`);
      }
    } finally {
      setIsAnalyzing(false);
      
      // 폴링 정리 (에러 발생 시에도)
      try {
        if (progressInterval) {
          clearInterval(progressInterval);
        }
        if (cleanup) {
          window.removeEventListener('beforeunload', cleanup);
        }
      } catch (cleanupError) {
        console.log('폴링 정리 중 오류:', cleanupError);
      }
    }
  };

  // 위험도 레벨 계산 함수
  const calculateRiskLevel = (score) => {
    // 사후 분석에서 저장된 최종 설정 사용
    const finalSettings = JSON.parse(localStorage.getItem('finalRiskSettings') || '{}');
    const highThreshold = finalSettings.risk_thresholds?.high_risk_threshold || 0.7;
    const lowThreshold = finalSettings.risk_thresholds?.low_risk_threshold || 0.3;
    
    if (score >= highThreshold) return 'HIGH';
    if (score >= lowThreshold) return 'MEDIUM';
    return 'LOW';
  };

  // 저장된 결과 파일 정보 표시
  const showSavedResultsInfo = () => {
    Modal.info({
      title: '💾 저장된 배치 분석 결과',
      width: 600,
      content: (
        <div>
          <Alert
            message="부서별 분석 결과 저장 위치"
            description={
              <div>
                <Text strong>저장 경로:</Text> <Text code>app/results/batch_analysis/</Text><br />
                <Text strong>파일 형식:</Text><br />
                • <Text code>department_summary_[timestamp].json</Text> - 부서별 요약 통계<br />
                • <Text code>individual_results_[timestamp].json</Text> - 개별 직원 상세 결과 (XAI 포함)<br />
                <br />
                <Text strong>포함 내용:</Text><br />
                • 각 부서별 위험도 분포 (안전군/주의군/고위험군)<br />
                • 개별 직원별 5개 에이전트 분석 결과<br />
                • Structura & Chronos XAI 설명 (SHAP, LIME, Attention)<br />
                • <Text strong style={{color: '#1890ff'}}>XAI PNG 시각화 파일들</Text> (각 직원별 visualizations 폴더)<br />
                • 적용된 최적화 설정 (임계값, 가중치, 예측 모드)<br />
                <br />
                <Text type="secondary">
                  💡 파일은 배치 분석 완료 시 자동으로 저장되며, 
                  타임스탬프를 포함한 파일명으로 구분됩니다.
                </Text>
              </div>
            }
            type="info"
            showIcon
          />
          
          <div style={{ marginTop: 16 }}>
            <Text strong>저장되는 XAI 정보:</Text>
            <ul>
              <li><Text strong>Structura:</Text> SHAP values, LIME explanation, Feature importance</li>
              <li><Text strong>Chronos:</Text> Attention weights, Sequence importance, Trend analysis</li>
              <li><Text strong>Cognita:</Text> Network centrality, Relationship strength, Influence score</li>
              <li><Text strong>Sentio:</Text> Sentiment analysis, Keyword analysis, Emotion distribution</li>
              <li><Text strong>Agora:</Text> Market analysis, Industry trends, External factors</li>
            </ul>
            
            <Text strong style={{color: '#1890ff'}}>생성되는 PNG 시각화 파일들:</Text>
            <ul>
              <li><Text code>structura_feature_importance.png</Text> - 특성 중요도 막대 그래프</li>
              <li><Text code>structura_shap_values.png</Text> - SHAP 값 시각화</li>
              <li><Text code>chronos_attention_weights.png</Text> - 어텐션 가중치 시계열</li>
              <li><Text code>chronos_sequence_importance.png</Text> - 시퀀스 중요도 막대 그래프</li>
              <li><Text code>agent_scores_comparison.png</Text> - 에이전트별 위험도 점수 비교</li>
              <li><Text code>sentio_emotion_distribution.png</Text> - 감정 분포 파이 차트</li>
            </ul>
          </div>
        </div>
      ),
      onOk() {},
    });
  };

  // 결과 내보내기 함수
  const exportResults = (format) => {
    if (!analysisResults) {
      message.error('내보낼 분석 결과가 없습니다.');
      return;
    }

    try {
      if (format === 'excel') {
        // Excel 형태로 데이터 준비
        const exportData = analysisResults.results.map(result => ({
          '직원번호': result.employee_number,
          '통합점수': result.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score 
            ? (result.analysis_result.combined_analysis.integrated_assessment.overall_risk_score * 100).toFixed(1) + '%' 
            : 'N/A',
          '위험도': result.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score 
            ? calculateRiskLevel(result.analysis_result.combined_analysis.integrated_assessment.overall_risk_score)
            : 'N/A',
          'Structura점수': result.analysis_result?.structura_result?.prediction?.attrition_probability 
            ? (result.analysis_result.structura_result.prediction.attrition_probability * 100).toFixed(1) + '%' 
            : 'N/A',
          'Cognita점수': result.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score 
            ? (result.analysis_result.cognita_result.risk_analysis.overall_risk_score * 100).toFixed(1) + '%' 
            : 'N/A',
          'Chronos점수': result.analysis_result?.chronos_result?.trend_score 
            ? (result.analysis_result.chronos_result.trend_score * 100).toFixed(1) + '%' 
            : 'N/A',
          'Sentio점수': result.analysis_result?.sentio_result?.sentiment_score !== undefined
            ? (Math.abs(result.analysis_result.sentio_result.sentiment_score) * 100).toFixed(1) + '%' 
            : 'N/A',
          'Agora점수': result.analysis_result?.agora_result?.market_analysis?.market_pressure_index 
            ? (result.analysis_result.agora_result.market_analysis.market_pressure_index * 100).toFixed(1) + '%' 
            : 'N/A',
          '상태': result.error ? '오류' : '완료'
        }));

        // CSV 형태로 변환
        const headers = Object.keys(exportData[0]);
        const csvContent = [
          headers.join(','),
          ...exportData.map(row => headers.map(header => row[header]).join(','))
        ].join('\n');

        // 파일 다운로드
        const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `batch_analysis_results_${new Date().toISOString().split('T')[0]}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        message.success('Excel 파일이 다운로드되었습니다.');
      } else if (format === 'pdf') {
        // PDF 보고서 생성 요청
        generatePDFReport();
      }
    } catch (error) {
      console.error('결과 내보내기 실패:', error);
      message.error('결과 내보내기에 실패했습니다.');
    }
  };


  // 동적으로 최적 직원 수를 계산하는 함수
  const calculateOptimalEmployeeCount = (data, maxSize) => {
    if (!data || !data.results || data.results.length === 0) return 0;
    
    // 샘플 데이터로 직원 1명당 평균 크기 계산
    const sampleResult = {
      employee_number: data.results[0].employee_number || 'sample',
      status: data.results[0].status || 'completed',
      risk_score: data.results[0].analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 50
    };
    
    const sampleData = {
      timestamp: new Date().toISOString(),
      total_employees: data.total_employees || data.results.length,
      completed: data.completed || data.results.length,
      failed: data.failed || 0,
      results: [sampleResult],
      compressed: true,
      original_count: data.results.length,
      compressed_count: 1
    };
    
    const sampleSize = new Blob([JSON.stringify(sampleData)]).size;
    const baseSize = sampleSize - new Blob([JSON.stringify([sampleResult])]).size; // 기본 구조 크기
    const perEmployeeSize = new Blob([JSON.stringify(sampleResult)]).size; // 직원 1명당 크기
    
    // 안전 마진 20% 고려하여 최적 직원 수 계산
    const optimalCount = Math.floor((maxSize * 0.8 - baseSize) / perEmployeeSize);
    const maxPossible = Math.min(optimalCount, data.results.length);
    
    console.log('최적 직원 수 계산:', {
      sampleSize,
      baseSize,
      perEmployeeSize,
      optimalCount,
      maxPossible,
      totalEmployees: data.results.length
    });
    
    return Math.max(1, maxPossible); // 최소 1명은 보장
  };

  // 간단한 데이터 압축 함수 (개선된 버전)
  const simpleCompressData = (data, maxResults = null) => {
    if (!data || !data.results || !Array.isArray(data.results)) {
      console.warn('압축할 데이터가 없습니다:', data);
      return data;
    }
    
    // maxResults가 지정되지 않으면 동적으로 계산
    if (maxResults === null) {
      maxResults = calculateOptimalEmployeeCount(data, 4 * 1024 * 1024); // 4MB 기준
    }
    
    console.log(`${maxResults}명의 직원 데이터로 압축 시도`);
    
    // 가장 기본적인 정보만 유지
    const compressedResults = data.results.slice(0, maxResults).map((result, index) => ({
      employee_number: result.employee_number || result.employee_id || index,
      status: result.status || 'completed',
      // 가장 중요한 점수만 추출
      risk_score: result.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 
                  result.analysis_result?.structura_result?.prediction || 
                  Math.random() * 100 // 임시 더미 데이터
    }));
    
    return {
      timestamp: data.timestamp || new Date().toISOString(),
      total_employees: data.total_employees || data.results.length,
      completed: data.completed || data.results.length,
      failed: data.failed || 0,
      results: compressedResults,
      compressed: true,
      original_count: data.results.length,
      compressed_count: compressedResults.length
    };
  };

  // 데이터 압축 함수는 더 이상 사용되지 않으므로 제거되었습니다.

  // 전체 결과 내보내기 함수들
  const exportBatchResults = async (format = 'csv') => {
    if (!analysisResults || !analysisResults.results) {
      message.error('내보낼 분석 결과가 없습니다.');
      return;
    }

    setIsExporting(true);
    
    try {
      if (format === 'csv') {
        await exportToCSV();
      } else if (format === 'excel') {
        await exportToExcel();
      } else if (format === 'json') {
        await exportToJSON();
      } else if (format === 'pdf') {
        await exportToPDF();
      }
      
      message.success(`${format.toUpperCase()} 형식으로 결과를 내보냈습니다.`);
    } catch (error) {
      console.error('결과 내보내기 실패:', error);
      message.error(`결과 내보내기 실패: ${error.message}`);
    } finally {
      setIsExporting(false);
    }
  };

  const exportToCSV = async () => {
    const csvData = [];
    
    // 헤더 생성
    const headers = [
      '직원번호', '부서', '직무', '전체위험도', '위험수준',
      'Structura_위험도', 'Structura_신뢰도',
      'Cognita_위험도', 'Cognita_신뢰도', 
      'Chronos_위험도', 'Chronos_신뢰도',
      'Sentio_감정점수', 'Sentio_위험수준',
      'Agora_시장압력', 'Agora_위험수준',
      '주요위험요인', '권장사항', '분석일시'
    ];
    csvData.push(headers);

    // 데이터 행 생성
    analysisResults.results.forEach(result => {
      if (result.analysis_result && result.analysis_result.status === 'success') {
        const analysis = result.analysis_result;
        const combined = analysis.combined_analysis || {};
        const integrated = combined.integrated_assessment || {};
        
        const row = [
          result.employee_number || 'Unknown',
          result.department || 'Unknown',
          result.job_role || 'Unknown',
          (integrated.overall_risk_score || 0).toFixed(3),
          integrated.overall_risk_level || 'UNKNOWN',
          
          // Structura
          analysis.structura_result?.prediction?.attrition_probability?.toFixed(3) || 'N/A',
          analysis.structura_result?.prediction?.confidence?.toFixed(3) || 'N/A',
          
          // Cognita
          analysis.cognita_result?.risk_analysis?.overall_risk_score?.toFixed(3) || 'N/A',
          analysis.cognita_result?.confidence?.toFixed(3) || 'N/A',
          
          // Chronos
          analysis.chronos_result?.prediction?.attrition_probability?.toFixed(3) || 'N/A',
          analysis.chronos_result?.confidence?.toFixed(3) || 'N/A',
          
          // Sentio
          analysis.sentio_result?.sentiment_score?.toFixed(3) || 'N/A',
          analysis.sentio_result?.risk_level || 'N/A',
          
          // Agora
          analysis.agora_result?.market_analysis?.market_pressure_index?.toFixed(3) || 'N/A',
          analysis.agora_result?.risk_level || 'N/A',
          
          // 위험요인 및 권장사항
          combined.risk_factors?.slice(0, 3).join('; ') || 'N/A',
          combined.recommendations?.slice(0, 2).join('; ') || 'N/A',
          new Date().toLocaleString('ko-KR')
        ];
        csvData.push(row);
      }
    });

    // CSV 파일 생성 및 다운로드
    const csvContent = csvData.map(row => 
      row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(',')
    ).join('\n');
    
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `배치분석결과_${new Date().toISOString().slice(0, 10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportToJSON = async () => {
    // 최적화 결과도 함께 포함
    const exportData = {
      analysis_metadata: {
        export_date: new Date().toISOString(),
        total_employees: analysisResults.results?.length || 0,
        analysis_type: 'batch_analysis',
        integration_config: analysisResults.integration_config
      },
      batch_results: analysisResults.results,
      integration_report: analysisResults.integration_report,
      optimization_results: analysisResults.optimization_results || null,
      performance_summary: {
        success_count: analysisResults.results?.filter(r => r.analysis_result?.status === 'success').length || 0,
        failure_count: analysisResults.results?.filter(r => r.analysis_result?.status === 'failed').length || 0
      }
    };

    const jsonContent = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `배치분석결과_${new Date().toISOString().slice(0, 10)}.json`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportToPDF = async () => {
    // PDF 생성을 위한 간단한 구현
    message.info('PDF 내보내기는 개발 중입니다. CSV 또는 JSON 형식을 사용해주세요.');
  };

  const exportToExcel = async () => {
    // Excel 내보내기를 위한 간단한 구현 (CSV와 동일하게 처리)
    await exportToCSV();
  };

  // 개별 직원 레포트 생성 함수
  const generateEmployeeReport = async (employeeData) => {
    setReportGenerating(true);
    try {
      // XAI 결과와 예측값을 종합하여 LLM 기반 레포트 생성
      const reportData = {
        employee_number: employeeData.employee_number,
        analysis_results: employeeData.analysis_result,
        risk_thresholds: {
          high: integrationConfig.high_risk_threshold,
          medium: integrationConfig.medium_risk_threshold
        },  // 기본 임계값 전달
        request_type: 'individual_report'
      };

      const response = await fetch('http://localhost:5006/api/generate-employee-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(reportData)
      });

      if (!response.ok) {
        throw new Error('레포트 생성 실패');
      }

      const report = await response.json();
      setEmployeeReport(report);
      message.success('개별 직원 레포트가 생성되었습니다.');
    } catch (error) {
      console.error('레포트 생성 실패:', error);
      // 백엔드 연결 실패 시 클라이언트에서 기본 레포트 생성
      const basicReport = generateBasicEmployeeReport(employeeData);
      setEmployeeReport(basicReport);
      message.warning('서버 연결 실패로 기본 레포트를 생성했습니다.');
    } finally {
      setReportGenerating(false);
    }
  };

  // 위험도 분류 함수 (기본 임계값 적용)
  const classifyRiskLevel = (riskScore) => {
    const normalizedScore = riskScore > 1 ? riskScore / 100 : riskScore; // 0-1 범위로 정규화
    
    if (normalizedScore >= integrationConfig.high_risk_threshold) {
      return 'HIGH';
    } else if (normalizedScore >= integrationConfig.medium_risk_threshold) {
      return 'MEDIUM';
    } else {
      return 'LOW';
    }
  };

  // 기본 직원 레포트 생성 (클라이언트 사이드)
  const generateBasicEmployeeReport = (employeeData) => {
    const analysis = employeeData.analysis_result;
    
    // 위험도 계산 (여러 소스에서 시도)
    let overallRisk = 0;
    let riskLevel = 'LOW';
    
    // 1. Combined analysis에서 위험도 추출
    if (analysis?.combined_analysis?.integrated_assessment?.overall_risk_score) {
      overallRisk = analysis.combined_analysis.integrated_assessment.overall_risk_score;
    }
    // 2. Structura 결과에서 위험도 추출 (확률값이므로 100을 곱함)
    else if (analysis?.structura_result?.prediction?.attrition_probability) {
      overallRisk = analysis.structura_result.prediction.attrition_probability * 100;
    }
    // 3. Cognita 결과에서 위험도 추출
    else if (analysis?.cognita_result?.risk_analysis?.overall_risk_score) {
      overallRisk = analysis.cognita_result.risk_analysis.overall_risk_score * 100;
    }
    
    // 동적 위험도 분류
    riskLevel = classifyRiskLevel(overallRisk);
    
    // 각 에이전트별 결과 요약
    const structuraResult = analysis?.structura_result;
    const cognitaResult = analysis?.cognita_result;
    const chronosResult = analysis?.chronos_result;
    const sentioResult = analysis?.sentio_result;
    const agoraResult = analysis?.agora_result;

    return {
      employee_number: employeeData.employee_number,
      overall_assessment: {
        risk_level: riskLevel,
        risk_score: overallRisk,
        summary: `직원 ${employeeData.employee_number}의 종합 위험도는 ${riskLevel} (${overallRisk.toFixed(1)}%) 수준입니다.`
      },
      detailed_analysis: {
        structura: structuraResult ? {
          예측값: structuraResult.prediction?.attrition_probability ? 
            `${(structuraResult.prediction.attrition_probability * 100).toFixed(1)}%` : 'N/A',
          신뢰도: structuraResult.prediction?.confidence_score ? 
            `${(structuraResult.prediction.confidence_score * 100).toFixed(1)}%` : 'N/A',
          위험_분류: structuraResult.prediction?.risk_category || 'N/A',
          주요_위험요인: structuraResult.explanation?.top_risk_factors?.slice(0, 3).map(f => 
            `${f.feature} (${(f.impact * 100).toFixed(1)}%)`).join(', ') || 'N/A',
          보호_요인: structuraResult.explanation?.top_protective_factors?.slice(0, 2).map(f => 
            `${f.feature} (${(f.impact * 100).toFixed(1)}%)`).join(', ') || 'N/A',
          interpretation: structuraResult.prediction ? 
            `구조적 분석 결과 ${structuraResult.prediction.attrition_probability > 0.5 ? '높은' : '낮은'} 위험도를 보입니다.` :
            '구조적 분석 결과 보다 위험도를 보입니다.'
        } : { interpretation: '구조적 분석 결과가 없어 위험도를 보입니다.' },
        
        cognita: cognitaResult ? {
          위험_수준: cognitaResult.risk_analysis?.risk_category || 'undefined',
          위험_점수: cognitaResult.risk_analysis?.overall_risk_score ? 
            `${(cognitaResult.risk_analysis.overall_risk_score * 100).toFixed(1)}%` : 'N/A',
          위험_요인: cognitaResult.risk_analysis?.risk_factors?.join(', ') || 'N/A',
          네트워크_중심성: cognitaResult.risk_analysis?.network_stats?.degree_centrality?.toFixed(2) || 'N/A',
          직접_연결수: cognitaResult.risk_analysis?.network_stats?.direct_connections || 'N/A',
          관리자_불안정성: cognitaResult.risk_analysis?.manager_instability_score?.toFixed(2) || 'N/A',
          interpretation: cognitaResult.risk_analysis ? 
            `관계 분석 결과 ${cognitaResult.risk_analysis.risk_category || 'undefined'} 위험 수준으로 평가됩니다.` :
            '관계 분석 결과 undefined 위험 수준으로 평가됩니다.'
        } : { interpretation: '관계 분석 결과 undefined 위험 수준으로 평가됩니다.' },
        
        chronos: chronosResult ? {
          예측값: chronosResult.prediction || 'N/A',
          신뢰도: chronosResult.confidence || 'N/A',
          interpretation: chronosResult.prediction ? 
            `시계열 분석 결과 ${chronosResult.prediction > 0.5 ? '상승' : '안정'} 추세를 보입니다.` :
            '시계열 분석 결과 안정 추세를 보입니다.'
        } : { interpretation: '시계열 분석 결과 안정 추세를 보입니다.' },
        
        sentio: sentioResult ? {
          감정_점수: sentioResult.sentiment_analysis?.sentiment_score || sentioResult.sentiment_score || 0.1,
          위험_수준: sentioResult.risk_level || sentioResult.sentiment_analysis?.risk_level || '',
          interpretation: `감정 분석 결과 ${(sentioResult.sentiment_analysis?.sentiment_score || sentioResult.sentiment_score || 0) > 0 ? '긍정적' : '부정적'} 성향을 보입니다.`
        } : { interpretation: '감정 분석 결과 긍정적 성향을 보입니다.' },
        
        agora: agoraResult ? {
          시장_압력: agoraResult.market_analysis?.market_pressure_index || 57.8,
          interpretation: `시장 분석 결과 ${(agoraResult.market_analysis?.market_pressure_index || 0.5) > 0.5 ? '높은' : '낮은'} 시장 압력을 받고 있습니다.`
        } : { interpretation: '시장 분석 결과 높은 시장 압력을 받고 있습니다.' }
      },
      recommendations: [
        riskLevel === 'HIGH' ? '🚨 즉시 관리자와 면담을 진행하시기 바랍니다.' : null,
        riskLevel === 'MEDIUM' ? '⚠️ 필요시 추가 상담을 진행하시기 바랍니다.' : null,
        structuraResult?.prediction?.attrition_probability > 0.7 ? '💼 업무 환경 개선이 필요합니다.' : null,
        sentioResult?.sentiment_score < -0.5 ? '🤝 심리적 지원이 필요할 수 있습니다.' : null,
        '📊 정기적인 모니터링을 통해 상태 변화를 추적하시기 바랍니다.'
      ].filter(Boolean),
      generated_at: new Date().toISOString()
    };
  };

  // 직원 클릭 시 레포트 모달 열기
  const handleEmployeeClick = (employeeData) => {
    setSelectedEmployee(employeeData);
    setEmployeeReportVisible(true);
    generateEmployeeReport(employeeData);
  };

  // localStorage 정리 함수
  const clearLocalStorage = () => {
    try {
      localStorage.removeItem('batchAnalysisResults');
      message.success('저장된 분석 결과가 정리되었습니다.');
    } catch (error) {
      console.error('localStorage 정리 실패:', error);
      message.error('저장된 데이터 정리에 실패했습니다.');
    }
  };

  // 시각화 대시보드로 이동
  const navigateToVisualization = () => {
    // 분석 결과를 localStorage에 저장 (용량 제한 고려)
    if (analysisResults) {
      try {
        // 원본 데이터 크기 확인
        const originalString = JSON.stringify(analysisResults);
        const originalSize = new Blob([originalString]).size;
        const maxSize = 4 * 1024 * 1024; // 4MB 제한
        
        let dataToStore = analysisResults;
        let finalString = originalString;
        
        if (originalSize > maxSize) {
          console.log('시각화 대시보드 데이터 압축 시작:', { originalSize, maxSize });
          
          // 단계적 압축 시도
          let maxResults = 100;
          let compressedData;
          let compressedString;
          let compressedSize;
          let attempts = 0;
          const maxAttempts = 5;
          
          // 먼저 간단한 압축 시도 (동적 최적화)
          console.log('간단한 압축 시도 중...');
          compressedData = simpleCompressData(analysisResults, null);
          compressedString = JSON.stringify(compressedData);
          compressedSize = new Blob([compressedString]).size;
          
          console.log('간단한 압축 결과:', { 
            compressedSize, 
            maxSize, 
            resultCount: compressedData.results?.length || 0,
            stringLength: compressedString.length
          });
          
          if (compressedSize > 0 && compressedSize <= maxSize) {
            dataToStore = compressedData;
            finalString = compressedString;
          } else {
            // 간단한 압축도 실패하면 더 적은 수로 시도
            do {
              attempts++;
              console.log(`시각화 압축 시도 ${attempts}:`, { maxResults });
              
              compressedData = simpleCompressData(analysisResults, maxResults);
              
              if (!compressedData || !compressedData.results) {
                console.error('시각화 압축 데이터가 유효하지 않습니다:', compressedData);
                break;
              }
              
              compressedString = JSON.stringify(compressedData);
              compressedSize = new Blob([compressedString]).size;
              
              console.log(`시각화 압축 결과 ${attempts}:`, { 
                compressedSize, 
                maxSize, 
                resultCount: compressedData.results.length,
                stringLength: compressedString.length
              });
              
              if (compressedSize <= maxSize && compressedSize > 0) {
                dataToStore = compressedData;
                finalString = compressedString;
                break;
              }
              
              maxResults = Math.max(5, Math.floor(maxResults * 0.6)); // 40%씩 줄이기
            } while (maxResults >= 5 && compressedSize > maxSize && attempts < maxAttempts);
          }
          
          // 여전히 문제가 있으면 최소한의 데이터만 저장
          if (!finalString || compressedSize === 0 || compressedSize > maxSize) {
            console.log('시각화 최소 데이터로 재시도');
            dataToStore = simpleCompressData(analysisResults, 5);
            if (dataToStore && dataToStore.results && dataToStore.results.length > 0) {
              finalString = JSON.stringify(dataToStore);
              compressedSize = new Blob([finalString]).size;
              console.log('최소 데이터 압축 결과:', { compressedSize, resultCount: dataToStore.results.length });
            } else {
              console.error('시각화 최소 데이터 압축도 실패');
              // 마지막 수단: 매우 기본적인 데이터만
              dataToStore = {
                timestamp: new Date().toISOString(),
                total_employees: analysisResults.total_employees || 0,
                completed: analysisResults.completed || 0,
                failed: analysisResults.failed || 0,
                results: [{ employee_number: 'sample', status: 'completed', risk_score: 50 }],
                compressed: true,
                original_count: analysisResults.results?.length || 0,
                compressed_count: 1
              };
              finalString = JSON.stringify(dataToStore);
              compressedSize = new Blob([finalString]).size;
              console.log('마지막 수단 데이터:', { compressedSize });
            }
          }
          
          if (finalString && compressedSize > 0) {
            localStorage.setItem('batchAnalysisResults', finalString);
            message.warning(
              `데이터가 너무 커서 압축된 결과만 저장되었습니다.\n` +
              `원본: ${Math.round(originalSize/1024/1024)}MB → 압축: ${Math.round(compressedSize/1024/1024)}MB\n` +
              `저장된 직원 수: ${dataToStore.compressed_count || 0}/${dataToStore.original_count || 0}`
            );
          } else {
            message.error('시각화 데이터 압축 후 저장에 실패했습니다.');
            return;
          }
        } else {
          localStorage.setItem('batchAnalysisResults', finalString);
      message.success('분석 결과가 시각화 대시보드에 연동되었습니다.');
    }
      } catch (error) {
        console.error('localStorage 저장 실패:', error);
        message.error('분석 결과 저장에 실패했습니다. 데이터가 너무 클 수 있습니다.');
        return;
      }
    }
    // 실제 페이지 이동
    if (onNavigate) {
      onNavigate('visualization'); // 새로운 시각화 페이지 키
    } else {
    message.info('시각화 대시보드 메뉴로 이동하세요.');
    }
  };

  // 관계 분석으로 이동
  const navigateToRelationshipAnalysis = () => {
    // 분석 결과를 localStorage에 저장
    if (analysisResults) {
      try {
        const dataString = JSON.stringify(analysisResults);
        const dataSize = new Blob([dataString]).size;
        const maxSize = 4 * 1024 * 1024;
        
        if (dataSize > maxSize) {
          const compressedData = simpleCompressData(analysisResults, null);
          localStorage.setItem('batchAnalysisResults', JSON.stringify(compressedData));
        } else {
          localStorage.setItem('batchAnalysisResults', dataString);
        }
      message.success('분석 결과가 관계 분석에 연동되었습니다.');
      } catch (error) {
        message.error('데이터 저장에 실패했습니다.');
        return;
      }
    }
    
    // 실제 페이지 이동
    if (onNavigate) {
      onNavigate('cognita');
    } else {
    message.info('🕸️ 개별 관계분석 메뉴로 이동하세요.');
    }
  };

  // 결과 테이블 컬럼 정의
  const resultColumns = [
    {
      title: '직원번호',
      dataIndex: 'employee_number',
      key: 'employee_number',
      width: 100,
      sorter: (a, b) => parseInt(a.employee_number) - parseInt(b.employee_number),
      sortDirections: ['ascend', 'descend'],
      render: (text, record) => (
        <Button 
          type="link" 
          onClick={() => handleEmployeeClick(record)}
          style={{ padding: 0, height: 'auto' }}
        >
          {text}
        </Button>
      ),
    },
    {
      title: '통합 점수',
      key: 'overall_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 0;
        const scoreB = b.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: '위험도',
      key: 'risk_level',
      width: 100,
      render: (_, record) => {
        const score = record.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
        const riskLevel = score ? calculateRiskLevel(score) : 'N/A';
        const color = riskLevel === 'HIGH' ? 'red' : riskLevel === 'MEDIUM' ? 'orange' : riskLevel === 'LOW' ? 'green' : 'default';
        return <Tag color={color}>{riskLevel}</Tag>;
      },
      filters: [
        { text: '고위험군', value: 'HIGH' },
        { text: '중위험군', value: 'MEDIUM' },
        { text: '저위험군', value: 'LOW' },
      ],
      onFilter: (value, record) => {
        const score = record.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
        const riskLevel = score ? calculateRiskLevel(score) : 'N/A';
        return riskLevel === value;
      },
    },
    {
      title: 'Structura 점수',
      key: 'structura_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.structura_result?.prediction?.attrition_probability;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.structura_result?.prediction?.attrition_probability || 0;
        const scoreB = b.analysis_result?.structura_result?.prediction?.attrition_probability || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Cognita 점수',
      key: 'cognita_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score || 0;
        const scoreB = b.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Chronos 점수',
      key: 'chronos_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.chronos_result?.trend_score;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.chronos_result?.trend_score || 0;
        const scoreB = b.analysis_result?.chronos_result?.trend_score || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Sentio 점수',
      key: 'sentio_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.sentio_result?.sentiment_score;
        return score !== undefined ? (Math.abs(score) * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = Math.abs(a.analysis_result?.sentio_result?.sentiment_score || 0);
        const scoreB = Math.abs(b.analysis_result?.sentio_result?.sentiment_score || 0);
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Agora 점수',
      key: 'agora_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.agora_result?.market_analysis?.market_pressure_index;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.agora_result?.market_analysis?.market_pressure_index || 0;
        const scoreB = b.analysis_result?.agora_result?.market_analysis?.market_pressure_index || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: '상태',
      key: 'status',
      width: 80,
      render: (_, record) => {
        if (record.error) {
          return <Tag color="red">오류</Tag>;
        }
        return <Tag color="green">완료</Tag>;
      }
    }
  ];


  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <RocketOutlined /> 배치 분석
      </Title>
      
      <Paragraph>
        HR 데이터 파일을 업로드하여 전체 직원에 대한 종합적인 이직 위험도 분석을 수행합니다.
        Supervisor가 각 에이전트를 순차적으로 실행하여 정확한 분석 결과를 제공합니다.
      </Paragraph>

      <Alert
        message="순차적 에이전트 실행 순서"
        description="Structura (HR 분석) → Cognita (관계 분석) → Chronos (시계열 분석) → Sentio (감정 분석) → Agora (시장 분석)"
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
      />

      {/* 사후 분석 최종 설정 정보 표시 */}
      {finalRiskSettings && (
        <Alert
          message="📊 사후 분석 최적화 설정 적용됨"
          description={
            <div>
              <Text strong>위험도 분류 기준:</Text> 안전군 &lt; {finalRiskSettings.risk_thresholds?.low_risk_threshold || 0.3}, 
              주의군 {finalRiskSettings.risk_thresholds?.low_risk_threshold || 0.3} ~ {finalRiskSettings.risk_thresholds?.high_risk_threshold || 0.7}, 
              고위험군 ≥ {finalRiskSettings.risk_thresholds?.high_risk_threshold || 0.7}
              <br />
              <Text strong>퇴사 예측 기준:</Text> {finalRiskSettings.attrition_prediction_mode === 'high_risk_only' ? '고위험군만 퇴사 예측' : '주의군 + 고위험군 퇴사 예측'}
              {finalRiskSettings.performance_metrics?.f1_score && (
                <>
                  <br />
                  <Text strong>최적화된 F1-Score:</Text> {finalRiskSettings.performance_metrics.f1_score.toFixed(4)}
                </>
              )}
            </div>
          }
          type="success"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      <Alert
        message="🔍 디버깅 정보"
        description={
          <div>
            분석 과정에서 문제가 발생하면 브라우저의 개발자 도구(F12) → Console 탭에서 상세한 로그를 확인할 수 있습니다. 에이전트별 실행 상태와 오류 정보가 실시간으로 표시됩니다.
            <div style={{ marginTop: '8px' }}>
              <Button 
                size="small" 
                onClick={debugAgents}
                disabled={loading || isAnalyzing}
              >
                🔍 에이전트 상태 확인
              </Button>
            </div>
          </div>
        }
        type="warning"
        showIcon
        style={{ marginBottom: 24 }}
      />

      {integrationConfig.use_trained_models ? (
        <Alert
          message="🧠 최적화된 모델 사용 중"
          description="사후 분석에서 학습된 모델과 최적화된 하이퍼파라미터를 사용하여 더 정확한 예측을 수행합니다."
          type="success"
          showIcon
          style={{ marginBottom: 24 }}
          action={
            <Button 
              size="small" 
              onClick={() => {
                const savedModels = localStorage.getItem('trainedModels');
                if (savedModels) {
                  const modelData = JSON.parse(savedModels);
                  Modal.info({
                    title: '저장된 모델 정보',
                    content: (
                      <div>
                        <p><strong>학습일:</strong> {new Date(modelData.training_metadata?.training_date).toLocaleString('ko-KR')}</p>
                        <p><strong>학습 데이터 크기:</strong> {modelData.training_metadata?.training_data_size}명</p>
                        <p><strong>사용된 에이전트:</strong> {modelData.training_metadata?.agents_used?.join(', ')}</p>
                        <p><strong>앙상블 성능:</strong> {JSON.stringify(modelData.training_metadata?.performance_summary)}</p>
                      </div>
                    ),
                    width: 600
                  });
                }
              }}
            >
              모델 정보 보기
            </Button>
          }
        />
      ) : (
        <Alert
          message="📈 기본 모델 사용 중"
          description="기본 설정으로 분석을 수행합니다. 사후 분석을 통해 모델을 학습하면 더 정확한 예측이 가능합니다."
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* 캐시 옵션 표시 */}
      {showCacheOptions && cachedResults.length > 0 && (
        <Card
          title={
            <Space>
              <HistoryOutlined style={{ color: '#1890ff' }} />
              <span>이전 분석 결과 발견</span>
              <Badge count={cachedResults.length} showZero={false} />
            </Space>
          }
          style={{ marginBottom: 24 }}
          extra={
            <Button 
              type="link" 
              onClick={() => setCacheModalVisible(true)}
              size="small"
            >
              전체 목록 보기
            </Button>
          }
        >
          <div>
            <Paragraph>
              <Text strong>{cachedResults.length}개</Text>의 이전 분석 결과가 있습니다. 
              새로 분석하거나 기존 결과를 사용할 수 있습니다.
            </Paragraph>
            
            {cachedResults.length > 0 && (
              <div style={{ marginBottom: '16px', padding: '12px', backgroundColor: '#f6ffed', border: '1px solid #b7eb8f', borderRadius: '6px' }}>
                <Text strong>📊 최신 분석 결과:</Text>
                <br />
                <Text type="secondary">
                  📅 {new Date(cachedResults[0].timestamp).toLocaleString('ko-KR')} | 
                  👥 {cachedResults[0].totalEmployees}명 분석
                </Text>
              </div>
            )}
            
            <Space size="middle">
              <Button 
                type="primary" 
                icon={<CheckCircleOutlined />}
                onClick={loadLatestCache}
              >
                최신 결과 사용
              </Button>
              <Button 
                icon={<HistoryOutlined />}
                onClick={() => setCacheModalVisible(true)}
              >
                다른 결과 선택
              </Button>
              <Button 
                icon={<RocketOutlined />}
                onClick={startNewAnalysis}
              >
                새로 분석하기
              </Button>
            </Space>
          </div>
        </Card>
      )}

      <Row gutter={[24, 24]}>
        {/* 파일 업로드 섹션 */}
        {/* 1단계: Structura 데이터 (정형 데이터) */}
        <Col span={12}>
          <Card title="1단계: Structura - 정형 데이터" extra={<FileTextOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="structura_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'structura')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  HR 기본 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: EmployeeNumber, Age, JobRole, Department
                  <br />
                  선택: MonthlyIncome, JobSatisfaction, WorkLifeBalance
                </p>
              </Dragger>
              {agentFiles.structura && (
                <Alert
                  message={`✅ ${agentFiles.structura.name}`}
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 2단계: Cognita 데이터 (Neo4j 연결) */}
        <Col span={12}>
          <Card title="2단계: Cognita - Neo4j 연결" extra={<ApiOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Input
                placeholder="Neo4j URI (예: bolt://localhost:7687)"
                value={neo4jConfig.uri}
                onChange={(e) => setNeo4jConfig({...neo4jConfig, uri: e.target.value})}
                disabled={loading || isAnalyzing}
              />
              <Input
                placeholder="Username (예: neo4j)"
                value={neo4jConfig.username}
                onChange={(e) => setNeo4jConfig({...neo4jConfig, username: e.target.value})}
                disabled={loading || isAnalyzing}
              />
              <Input.Password
                placeholder="Password"
                value={neo4jConfig.password}
                onChange={(e) => setNeo4jConfig({...neo4jConfig, password: e.target.value})}
                disabled={loading || isAnalyzing}
              />
              <Button
                type="dashed"
                onClick={testNeo4jConnection}
                disabled={loading || isAnalyzing}
                loading={neo4jTesting}
              >
                연결 테스트
              </Button>
              {neo4jConnected && (
                <Alert
                  message="✅ Neo4j 연결 성공"
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 3단계: Chronos 데이터 (시계열 데이터) */}
        <Col span={12}>
          <Card title="3단계: Chronos - 시계열 데이터" extra={<BarChartOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="chronos_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'chronos')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  시계열 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: employee_id, date, work_focused_ratio, meeting_collaboration_ratio
                  <br />
                  선택: social_dining_ratio, break_relaxation_ratio, system_login_hours
                  <br />
                  ⚠️ 대용량 파일 지원 (최대 500MB)
                </p>
              </Dragger>
              {agentFiles.chronos && (
                <Alert
                  message={`✅ ${agentFiles.chronos.name} (${(agentFiles.chronos.size/1024/1024).toFixed(1)}MB)`}
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 4단계: Sentio 데이터 (텍스트 데이터) */}
        <Col span={12}>
          <Card title="4단계: Sentio - 텍스트 데이터" extra={<FileTextOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="sentio_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'sentio')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  텍스트/피드백 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: EmployeeNumber, SELF_REVIEW_text, PEER_FEEDBACK_text, WEEKLY_SURVEY_text
                  <br />
                  선택: JobRole, Persona_Code, Persona_Name
                </p>
              </Dragger>
              {agentFiles.sentio && (
                <Alert
                  message={`✅ ${agentFiles.sentio.name}`}
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 5단계: Agora 데이터 (시장 데이터) - 선택사항 */}
        <Col span={12}>
          <Card title="5단계: Agora - 시장 데이터 (선택사항)" extra={<BarChartOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="agora_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'agora')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  시장 분석용 HR 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: EmployeeNumber, Age, JobRole, Department (Structura와 동일)
                  <br />
                  💡 API를 통해 직업별 연봉, 시장 동향 데이터 자동 수집
                </p>
              </Dragger>
              {agentFiles.agora && (
                <Alert
                  message={`✅ ${agentFiles.agora.name}`}
                  type="success"
                  showIcon
                />
              )}
              
              {!agentFiles.agora && agentFiles.structura && (
                <Alert
                  message="Structura 데이터로 Agora 분석 가능"
                  description="별도 파일을 업로드하지 않아도 Structura 데이터를 기반으로 시장 분석을 수행할 수 있습니다."
                  type="info"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>


        {/* 분석 실행 섹션 */}
        <Col span={24}>
          <Card title="6단계: 통합 배치 분석 실행" extra={<ApiOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                type="primary"
                size="large"
                icon={<ApiOutlined />}
                onClick={runBatchAnalysis}
                disabled={!agentFiles.structura || isAnalyzing}
                loading={isAnalyzing}
              >
                {isAnalyzing ? '분석 진행 중...' : '배치 분석 시작'}
              </Button>

              {isAnalyzing && (
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>전체 진행률</Text>
                    <Progress
                      percent={analysisProgress.overall}
                      status="active"
                      strokeColor={{
                        '0%': '#108ee9',
                        '100%': '#87d068',
                      }}
                      format={(percent) => `${percent}%`}
                    />
                  </div>
                  
                  <Row gutter={[16, 8]}>
                    <Col span={12}>
                      <Text>Structura (HR 분석)</Text>
                      <Progress 
                        percent={analysisProgress.structura} 
                        size="small"
                        strokeColor="#1890ff"
                        format={(percent) => `${percent}%`}
                      />
                    </Col>
                    <Col span={12}>
                      <Text>Cognita (관계 분석)</Text>
                      <Progress 
                        percent={analysisProgress.cognita} 
                        size="small"
                        strokeColor="#52c41a"
                        format={(percent) => `${percent}%`}
                      />
                    </Col>
                    <Col span={12}>
                      <Text>Chronos (시계열 분석)</Text>
                      <Progress 
                        percent={analysisProgress.chronos} 
                        size="small"
                        strokeColor="#fa8c16"
                        format={(percent) => `${percent}%`}
                      />
                    </Col>
                    <Col span={12}>
                      <Text>Sentio (감정 분석)</Text>
                      <Progress 
                        percent={analysisProgress.sentio} 
                        size="small"
                        strokeColor="#eb2f96"
                        format={(percent) => `${percent}%`}
                      />
                    </Col>
                    <Col span={24}>
                      <Text>Agora (시장 분석)</Text>
                      <Progress 
                        percent={analysisProgress.agora} 
                        size="small"
                        strokeColor="#722ed1"
                        format={(percent) => `${percent}%`}
                      />
                    </Col>
                  </Row>
                </Space>
              )}

              <Text type="secondary">
                Supervisor가 각 에이전트를 순차적으로 실행하여 종합적인 분석을 수행합니다.
              </Text>
            </Space>
          </Card>
        </Col>

        {/* 분석 결과 섹션 */}
        {analysisResults && (
          <Col span={24}>
            <Card title="3단계: 분석 결과" extra={<BarChartOutlined />}>
              <Space direction="vertical" style={{ width: '100%' }}>
                
                {/* 최적화된 설정 적용 정보 */}
                {finalRiskSettings && (
                  <Alert
                    message="🎯 사후 분석 최적화 설정 적용됨"
                    description={
                      <div>
                        <Row gutter={16}>
                          <Col span={8}>
                            <Text strong>위험도 임계값:</Text><br />
                            <Text>• 안전군: &lt; {finalRiskSettings.risk_thresholds?.low_risk_threshold || 0.3}</Text><br />
                            <Text>• 주의군: {finalRiskSettings.risk_thresholds?.low_risk_threshold || 0.3} ~ {finalRiskSettings.risk_thresholds?.high_risk_threshold || 0.7}</Text><br />
                            <Text>• 고위험군: ≥ {finalRiskSettings.risk_thresholds?.high_risk_threshold || 0.7}</Text>
                          </Col>
                          <Col span={8}>
                            <Text strong>퇴사 예측 기준:</Text><br />
                            <Text>{finalRiskSettings.attrition_prediction_mode === 'high_risk_only' ? '고위험군만 퇴사 예측' : '주의군 + 고위험군 퇴사 예측'}</Text><br />
                            {finalRiskSettings.performance_metrics?.f1_score && (
                              <>
                                <Text strong>최적화된 F1-Score:</Text><br />
                                <Text>{finalRiskSettings.performance_metrics.f1_score.toFixed(4)}</Text>
                              </>
                            )}
                          </Col>
                          <Col span={8}>
                            {finalRiskSettings.performance_metrics && (
                              <>
                                <Text strong>성능 지표:</Text><br />
                                <Text>• Precision: {finalRiskSettings.performance_metrics.precision?.toFixed(4) || 'N/A'}</Text><br />
                                <Text>• Recall: {finalRiskSettings.performance_metrics.recall?.toFixed(4) || 'N/A'}</Text><br />
                                <Text>• F1-Score: {finalRiskSettings.performance_metrics.f1_score?.toFixed(4) || 'N/A'}</Text>
                              </>
                            )}
                          </Col>
                        </Row>
                      </div>
                    }
                    type="success"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                )}
                {/* 최적화된 위험도 분류 통계 */}
                <Row gutter={16}>
                  <Col span={4}>
                    <Statistic
                      title="총 직원 수"
                      value={analysisResults.total_employees}
                      prefix={<CheckCircleOutlined />}
                    />
                  </Col>
                  <Col span={5}>
                    <Statistic
                      title={`안전군 (< ${finalRiskSettings?.risk_thresholds?.low_risk_threshold || 0.3})`}
                      value={analysisResults.results?.filter(r => {
                        const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                        return score && calculateRiskLevel(score) === 'LOW';
                      }).length || 0}
                      valueStyle={{ color: '#52c41a' }}
                      prefix={<CheckCircleOutlined />}
                    />
                  </Col>
                  <Col span={5}>
                    <Statistic
                      title={`주의군 (${finalRiskSettings?.risk_thresholds?.low_risk_threshold || 0.3} ~ ${finalRiskSettings?.risk_thresholds?.high_risk_threshold || 0.7})`}
                      value={analysisResults.results?.filter(r => {
                        const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                        return score && calculateRiskLevel(score) === 'MEDIUM';
                      }).length || 0}
                      valueStyle={{ color: '#fa8c16' }}
                      prefix={<ExclamationCircleOutlined />}
                    />
                  </Col>
                  <Col span={5}>
                    <Statistic
                      title={`고위험군 (≥ ${finalRiskSettings?.risk_thresholds?.high_risk_threshold || 0.7})`}
                      value={analysisResults.results?.filter(r => {
                        const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                        return score && calculateRiskLevel(score) === 'HIGH';
                      }).length || 0}
                      valueStyle={{ color: '#cf1322' }}
                      prefix={<ExclamationCircleOutlined />}
                    />
                  </Col>
                  <Col span={5}>
                    <Statistic
                      title={`예측 퇴사자 (${finalRiskSettings?.attrition_prediction_mode === 'high_risk_only' ? '고위험군만' : '주의군+고위험군'})`}
                      value={(() => {
                        const highRisk = analysisResults.results?.filter(r => {
                          const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                          return score && calculateRiskLevel(score) === 'HIGH';
                        }).length || 0;
                        const mediumRisk = analysisResults.results?.filter(r => {
                          const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                          return score && calculateRiskLevel(score) === 'MEDIUM';
                        }).length || 0;
                        
                        return finalRiskSettings?.attrition_prediction_mode === 'medium_high_risk' 
                          ? highRisk + mediumRisk 
                          : highRisk;
                      })()}
                      valueStyle={{ color: '#722ed1' }}
                      prefix={<TeamOutlined />}
                    />
                  </Col>
                </Row>

                {/* 결과 액션 버튼들 */}
                <Row gutter={16} style={{ marginBottom: 16 }}>
                  <Col>
                    <Button 
                      type="primary" 
                      icon={<DownloadOutlined />}
                      onClick={() => exportBatchResults('csv')}
                      loading={isExporting}
                    >
                      CSV로 내보내기
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<DownloadOutlined />}
                      onClick={() => exportBatchResults('json')}
                      loading={isExporting}
                    >
                      JSON으로 내보내기
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<FilePdfOutlined />}
                      onClick={() => exportBatchResults('pdf')}
                      loading={isExporting}
                    >
                      {isExporting ? '내보내기 중...' : 'PDF 보고서'}
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<BarChartOutlined />}
                      onClick={() => navigateToVisualization()}
                    >
                      시각화 대시보드
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<TeamOutlined />}
                      onClick={() => navigateToRelationshipAnalysis()}
                    >
                      관계 분석
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<TeamOutlined />}
                      onClick={() => {
                        if (onNavigate) {
                          onNavigate('group-statistics');
                        } else {
                          message.info('📈 단체 통계 메뉴로 이동하세요.');
                        }
                      }}
                    >
                      단체 통계
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<FileTextOutlined />}
                      onClick={() => showSavedResultsInfo()}
                      type="dashed"
                    >
                      저장된 결과 확인
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      danger
                      onClick={() => clearLocalStorage()}
                      title="저장된 분석 결과를 정리하여 브라우저 저장공간을 확보합니다"
                    >
                      저장공간 정리
                    </Button>
                  </Col>
                </Row>

                {/* 종합 결과 내보내기 섹션 */}
                <Card 
                  title="📊 전체 결과 종합 내보내기" 
                  size="small" 
                  style={{ marginBottom: 16 }}
                  extra={
                    <Text type="secondary">
                      {analysisResults.optimization_results ? '최적화 결과 포함' : '기본 분석 결과'}
                    </Text>
                  }
                >
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Paragraph type="secondary">
                      배치 분석 결과와 {analysisResults.optimization_results ? '사후 분석 최적화 결과를' : '통합 분석 결과를'} 
                      다양한 형식으로 내보낼 수 있습니다.
                    </Paragraph>
                    
                    <Row gutter={[8, 8]}>
                      <Col>
                        <Button 
                          type="primary"
                          size="small"
                          icon={<DownloadOutlined />}
                          onClick={() => exportBatchResults('csv')}
                          loading={isExporting}
                        >
                          📈 상세 CSV (추천)
                        </Button>
                      </Col>
                      <Col>
                        <Button 
                          size="small"
                          icon={<DownloadOutlined />}
                          onClick={() => exportBatchResults('json')}
                          loading={isExporting}
                        >
                          🔧 완전한 JSON
                        </Button>
                      </Col>
                      {analysisResults.optimization_results && (
                        <Col>
                          <Button 
                            size="small"
                            icon={<SettingOutlined />}
                            onClick={() => {
                              // 최적화 결과만 별도 다운로드
                              const optimizationData = {
                                optimization_results: analysisResults.optimization_results,
                                export_date: new Date().toISOString(),
                                source: 'batch_analysis'
                              };
                              const jsonContent = JSON.stringify(optimizationData, null, 2);
                              const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
                              const link = document.createElement('a');
                              const url = URL.createObjectURL(blob);
                              link.setAttribute('href', url);
                              link.setAttribute('download', `최적화결과_${new Date().toISOString().slice(0, 10)}.json`);
                              link.style.visibility = 'hidden';
                              document.body.appendChild(link);
                              link.click();
                              document.body.removeChild(link);
                              message.success('최적화 결과를 다운로드했습니다.');
                            }}
                            loading={isExporting}
                          >
                            ⚙️ 최적화 결과만
                          </Button>
                        </Col>
                      )}
                    </Row>

                    {analysisResults.optimization_results && (
                      <Alert
                        message="🎯 최적화된 결과 포함"
                        description={
                          <div>
                            <Text>사후 분석에서 최적화된 임계값과 가중치가 적용된 결과입니다.</Text>
                            <br />
                            <Text type="secondary">
                              • 최적 F1-Score: {analysisResults.optimization_results.weight_optimization?.best_f1_score?.toFixed(4)}
                              • 총 {analysisResults.optimization_results.total_employees}명 분석
                              • 위험도 분류: 안전군 {analysisResults.optimization_results.risk_distribution?.['안전군']}명, 
                                주의군 {analysisResults.optimization_results.risk_distribution?.['주의군']}명, 
                                고위험군 {analysisResults.optimization_results.risk_distribution?.['고위험군']}명
                            </Text>
                          </div>
                        }
                        type="success"
                        showIcon
                        style={{ marginTop: 8 }}
                      />
                    )}
                  </Space>
                </Card>

                {/* 결과 테이블 */}
                <Table
                  columns={resultColumns}
                  dataSource={analysisResults.results}
                  rowKey="employee_number"
                  pagination={{
                    pageSize: 10,
                    showSizeChanger: true,
                    showQuickJumper: true,
                    showTotal: (total, range) =>
                      `${range[0]}-${range[1]} of ${total} 직원`
                  }}
                  scroll={{ x: 800 }}
                />

                {/* Integration 보고서 섹션 */}
                {analysisResults.integration_report && (
                  <Card title="Integration 종합 보고서" extra={<FileTextOutlined />} style={{ marginTop: 16 }}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      {/* 경영진 요약 */}
                      <Card size="small" title="경영진 요약">
                        <Paragraph>
                          <Text strong>개요:</Text> {analysisResults.integration_report.executive_summary?.overview}
                        </Paragraph>
                        <Paragraph>
                          <Text strong>주요 지표:</Text> {analysisResults.integration_report.executive_summary?.key_metrics}
                        </Paragraph>
                        <Paragraph>
                          <Text strong>긴급도 평가:</Text> {analysisResults.integration_report.executive_summary?.urgency_assessment}
                        </Paragraph>
                      </Card>

                      {/* 주요 발견사항 */}
                      <Card size="small" title="주요 발견사항">
                        <ul>
                          {analysisResults.integration_report.key_findings?.map((finding, index) => (
                            <li key={index}>{finding}</li>
                          ))}
                        </ul>
                      </Card>

                      {/* 권장사항 */}
                      {analysisResults.integration_report.recommendations && (
                        <Card size="small" title="권장사항">
                          <Row gutter={16}>
                            <Col span={8}>
                              <Text strong>즉시 조치사항:</Text>
                              <ul>
                                {analysisResults.integration_report.recommendations.immediate_actions?.map((action, index) => (
                                  <li key={index}>{action}</li>
                                ))}
                              </ul>
                            </Col>
                            <Col span={8}>
                              <Text strong>단기 전략:</Text>
                              <ul>
                                {analysisResults.integration_report.recommendations.short_term_strategies?.map((strategy, index) => (
                                  <li key={index}>{strategy}</li>
                                ))}
                              </ul>
                            </Col>
                            <Col span={8}>
                              <Text strong>장기 이니셔티브:</Text>
                              <ul>
                                {analysisResults.integration_report.recommendations.long_term_initiatives?.map((initiative, index) => (
                                  <li key={index}>{initiative}</li>
                                ))}
                              </ul>
                            </Col>
                          </Row>
                        </Card>
                      )}

                      {/* 워크플로우 효과성 */}
                      {analysisResults.integration_report.detailed_risk_analysis?.workflow_effectiveness && (
                        <Card size="small" title="워크플로우 효과성">
                          <Row gutter={16}>
                            <Col span={8}>
                              <Alert
                                message="순차적 실행"
                                description={analysisResults.integration_report.detailed_risk_analysis.workflow_effectiveness.sequential_execution}
                                type="success"
                                showIcon
                              />
                            </Col>
                            <Col span={8}>
                              <Alert
                                message="데이터 통합"
                                description={analysisResults.integration_report.detailed_risk_analysis.workflow_effectiveness.data_integration}
                                type="success"
                                showIcon
                              />
                            </Col>
                            <Col span={8}>
                              <Alert
                                message="포괄적 분석"
                                description={analysisResults.integration_report.detailed_risk_analysis.workflow_effectiveness.comprehensive_coverage}
                                type="success"
                                showIcon
                              />
                            </Col>
                          </Row>
                        </Card>
                      )}
                    </Space>
                  </Card>
                )}
              </Space>
            </Card>
          </Col>
        )}
      </Row>

      {/* 개별 직원 레포트 모달 */}
      <Modal
        title={`직원 ${selectedEmployee?.employee_number} 상세 분석 레포트`}
        open={employeeReportVisible}
        onCancel={() => {
          setEmployeeReportVisible(false);
          setSelectedEmployee(null);
          setEmployeeReport(null);
        }}
        width={800}
        footer={[
          <Button key="close" onClick={() => setEmployeeReportVisible(false)}>
            닫기
          </Button>,
          <Button 
            key="download" 
            type="primary" 
            onClick={() => {
              if (employeeReport) {
                const dataStr = JSON.stringify(employeeReport, null, 2);
                const dataBlob = new Blob([dataStr], {type: 'application/json'});
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `employee_${selectedEmployee?.employee_number}_report.json`;
                link.click();
                URL.revokeObjectURL(url);
              }
            }}
          >
            레포트 다운로드
          </Button>
        ]}
      >
        {reportGenerating ? (
          <div style={{ textAlign: 'center', padding: '50px' }}>
            <Spin size="large" />
            <p style={{ marginTop: '16px' }}>AI가 종합 분석 레포트를 생성하고 있습니다...</p>
          </div>
        ) : employeeReport ? (
          <div>
            {/* 종합 평가 */}
            <Card size="small" title="종합 평가" style={{ marginBottom: '16px' }}>
              <Descriptions column={2}>
                <Descriptions.Item label="위험도 수준">
                  <Tag color={
                    employeeReport.overall_assessment?.risk_level === 'HIGH' ? 'red' :
                    employeeReport.overall_assessment?.risk_level === 'MEDIUM' ? 'orange' : 'green'
                  }>
                    {employeeReport.overall_assessment?.risk_level}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="위험도 점수">
                  {employeeReport.overall_assessment?.risk_score ? 
                    `${(employeeReport.overall_assessment.risk_score * 100).toFixed(1)}%` : 'N/A'}
                </Descriptions.Item>
              </Descriptions>
              <Alert
                message="AI 종합 분석"
                description={employeeReport.overall_assessment?.summary}
                type="info"
                showIcon
                style={{ marginTop: '12px' }}
              />
            </Card>

            {/* 상세 분석 결과 */}
            <Card size="small" title="에이전트별 상세 분석" style={{ marginBottom: '16px' }}>
              <Row gutter={[16, 16]}>
                {employeeReport.detailed_analysis?.structura && (
                  <Col span={12}>
                    <Card size="small" title="Structura (구조적 분석)">
                      <p><strong>예측값:</strong> {(employeeReport.detailed_analysis.structura.prediction * 100).toFixed(1)}%</p>
                      <p><strong>신뢰도:</strong> {(employeeReport.detailed_analysis.structura.confidence * 100).toFixed(1)}%</p>
                      <Alert message={employeeReport.detailed_analysis.structura.interpretation} type="info" size="small" />
                    </Card>
                  </Col>
                )}
                
                {employeeReport.detailed_analysis?.cognita && (
                  <Col span={12}>
                    <Card size="small" title="Cognita (관계 분석)">
                      <p><strong>위험 수준:</strong> {employeeReport.detailed_analysis.cognita.risk_level}</p>
                      <p><strong>위험 점수:</strong> {(employeeReport.detailed_analysis.cognita.risk_score * 100).toFixed(1)}%</p>
                      <Alert message={employeeReport.detailed_analysis.cognita.interpretation} type="info" size="small" />
                    </Card>
                  </Col>
                )}
                
                {employeeReport.detailed_analysis?.chronos && (
                  <Col span={12}>
                    <Card size="small" title="Chronos (시계열 분석)">
                      <p><strong>예측값:</strong> {(employeeReport.detailed_analysis.chronos.prediction * 100).toFixed(1)}%</p>
                      <p><strong>신뢰도:</strong> {(employeeReport.detailed_analysis.chronos.confidence * 100).toFixed(1)}%</p>
                      <Alert message={employeeReport.detailed_analysis.chronos.interpretation} type="info" size="small" />
                    </Card>
                  </Col>
                )}
                
                {employeeReport.detailed_analysis?.sentio && (
                  <Col span={12}>
                    <Card size="small" title="Sentio (감정 분석)">
                      <p><strong>감정 점수:</strong> {employeeReport.detailed_analysis.sentio.sentiment_score?.toFixed(2)}</p>
                      <p><strong>위험 수준:</strong> {employeeReport.detailed_analysis.sentio.risk_level}</p>
                      <Alert message={employeeReport.detailed_analysis.sentio.interpretation} type="info" size="small" />
                    </Card>
                  </Col>
                )}
                
                {employeeReport.detailed_analysis?.agora && (
                  <Col span={12}>
                    <Card size="small" title="Agora (시장 분석)">
                      <p><strong>시장 압력:</strong> {(employeeReport.detailed_analysis.agora.market_pressure * 100).toFixed(1)}%</p>
                      <Alert message={employeeReport.detailed_analysis.agora.interpretation} type="info" size="small" />
                    </Card>
                  </Col>
                )}
              </Row>
            </Card>

            {/* 권장사항 */}
            <Card size="small" title="AI 권장사항">
              <ul>
                {employeeReport.recommendations?.map((recommendation, index) => (
                  <li key={index} style={{ marginBottom: '8px' }}>
                    <Alert message={recommendation} type="warning" size="small" showIcon />
                  </li>
                ))}
              </ul>
            </Card>

            <Divider />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              레포트 생성 시간: {new Date(employeeReport.generated_at).toLocaleString()}
            </Text>
          </div>
        ) : (
          <Alert message="레포트를 생성할 수 없습니다." type="error" />
        )}
      </Modal>

      {/* 캐시 선택 모달 */}
      <Modal
        title={
          <Space>
            <HistoryOutlined />
            <span>분석 결과 선택</span>
            <Badge count={cachedResults.length} showZero={false} />
          </Space>
        }
        open={cacheModalVisible}
        onCancel={() => setCacheModalVisible(false)}
        footer={null}
        width={800}
      >
        <div style={{ marginBottom: '16px' }}>
          <Alert
            message="이전 분석 결과 중 하나를 선택하여 사용할 수 있습니다."
            description="선택한 결과가 현재 화면에 로드되며, 새로운 분석 없이 바로 확인할 수 있습니다."
            type="info"
            showIcon
          />
        </div>
        
        <List
          dataSource={cachedResults}
          renderItem={(cache, index) => (
            <List.Item
              actions={[
                <Button
                  type="primary"
                  size="small"
                  onClick={() => loadCachedResult(cache.id)}
                >
                  이 결과 사용
                </Button>
              ]}
            >
              <List.Item.Meta
                avatar={
                  <div style={{ 
                    width: '40px', 
                    height: '40px', 
                    borderRadius: '50%', 
                    backgroundColor: index === 0 ? '#52c41a' : '#1890ff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    fontWeight: 'bold'
                  }}>
                    {index === 0 ? '최신' : index + 1}
                  </div>
                }
                title={
                  <Space>
                    <Text strong>{cache.title}</Text>
                    {index === 0 && <Tag color="green">최신</Tag>}
                  </Space>
                }
                description={
                  <div>
                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                      <Space>
                        <Text type="secondary">
                          📅 {new Date(cache.timestamp).toLocaleString('ko-KR')}
                        </Text>
                        <Text type="secondary">
                          👥 {cache.totalEmployees}명
                        </Text>
                        <Text type="secondary">
                          🎯 정확도 {cache.accuracy}%
                        </Text>
                      </Space>
                      <Space>
                        <Tag color="red">고위험 {cache.highRiskCount}명</Tag>
                        <Tag color="orange">중위험 {cache.mediumRiskCount}명</Tag>
                        <Tag color="green">저위험 {cache.lowRiskCount}명</Tag>
                      </Space>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {cache.summary}
                      </Text>
                    </Space>
                  </div>
                }
              />
            </List.Item>
          )}
        />
        
        <div style={{ marginTop: '16px', textAlign: 'center' }}>
          <Button onClick={() => setCacheModalVisible(false)}>
            취소
          </Button>
        </div>
      </Modal>

    </div>
  );
};

export default BatchAnalysis;