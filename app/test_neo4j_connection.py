#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j 연결 테스트 스크립트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_neo4j_direct():
    """Neo4j 직접 연결 테스트"""
    print("=== Neo4j 직접 연결 테스트 ===")
    
    try:
        from neo4j import GraphDatabase
        
        # 연결 정보
        uri = "bolt://44.212.67.74:7687"
        username = "neo4j"
        password = "legs-augmentations-cradle"
        
        print(f"연결 시도: {uri}")
        print(f"사용자: {username}")
        
        # 드라이버 생성
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # 연결 테스트
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            
            if record and record['test'] == 1:
                print("✅ Neo4j 직접 연결 성공!")
                
                # 데이터베이스 정보 확인
                node_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = node_result.single()['node_count']
                
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = rel_result.single()['rel_count']
                
                print(f"📊 노드 수: {node_count:,}")
                print(f"📊 관계 수: {rel_count:,}")
                
                # 직원 데이터 확인
                emp_result = session.run("MATCH (e:Employee) RETURN count(e) as emp_count LIMIT 1")
                emp_record = emp_result.single()
                if emp_record:
                    emp_count = emp_record['emp_count']
                    print(f"👥 직원 수: {emp_count:,}")
                
                return True
            else:
                print("❌ Neo4j 연결 테스트 실패")
                return False
                
        driver.close()
        
    except Exception as e:
        print(f"❌ Neo4j 연결 오류: {e}")
        return False

def test_cognita_module():
    """Cognita 모듈 import 테스트"""
    print("\n=== Cognita 모듈 테스트 ===")
    
    try:
        from Cognita.cognita_flask_backend import Neo4jManager, CognitaRiskAnalyzer
        print("✅ Cognita 모듈 import 성공")
        
        # Neo4jManager 테스트
        uri = "bolt://44.212.67.74:7687"
        username = "neo4j"
        password = "legs-augmentations-cradle"
        
        manager = Neo4jManager(uri, username, password)
        print("✅ Neo4jManager 생성 성공")
        
        # 연결 테스트
        with manager.driver.session() as session:
            result = session.run("RETURN 1 as test")
            if result.single()['test'] == 1:
                print("✅ Neo4jManager 연결 성공")
                
                # CognitaRiskAnalyzer 테스트
                analyzer = CognitaRiskAnalyzer(manager)
                print("✅ CognitaRiskAnalyzer 생성 성공")
                
                return True
        
        manager.close()
        
    except Exception as e:
        print(f"❌ Cognita 모듈 테스트 실패: {e}")
        return False

def test_master_server_import():
    """마스터 서버 import 테스트"""
    print("\n=== 마스터 서버 Import 테스트 ===")
    
    try:
        from agentic_master_server import WorkerAgentManager
        print("✅ WorkerAgentManager import 성공")
        
        # 워커 매니저 생성 테스트
        manager = WorkerAgentManager()
        print("✅ WorkerAgentManager 생성 성공")
        
        # 워커 상태 확인
        workers = list(manager.workers.keys())
        print(f"📊 초기화된 워커들: {workers}")
        
        if 'cognita' in workers:
            print("✅ Cognita 워커 초기화됨")
            cognita_status = manager.workers['cognita']['status']
            print(f"📊 Cognita 상태: {cognita_status.status}")
            if cognita_status.error_message:
                print(f"⚠️ Cognita 오류: {cognita_status.error_message}")
        else:
            print("❌ Cognita 워커 초기화 실패")
        
        return True
        
    except Exception as e:
        print(f"❌ 마스터 서버 import 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Neo4j 연결 문제 진단 시작...\n")
    
    # 1. Neo4j 직접 연결 테스트
    neo4j_ok = test_neo4j_direct()
    
    # 2. Cognita 모듈 테스트
    cognita_ok = test_cognita_module()
    
    # 3. 마스터 서버 테스트
    master_ok = test_master_server_import()
    
    print("\n=== 진단 결과 ===")
    print(f"Neo4j 직접 연결: {'✅' if neo4j_ok else '❌'}")
    print(f"Cognita 모듈: {'✅' if cognita_ok else '❌'}")
    print(f"마스터 서버: {'✅' if master_ok else '❌'}")
    
    if all([neo4j_ok, cognita_ok, master_ok]):
        print("\n🎉 모든 테스트 통과! 배치 분석에서 Neo4j 연결이 가능해야 합니다.")
    else:
        print("\n⚠️ 일부 테스트 실패. 위 결과를 확인하여 문제를 해결해주세요.")
