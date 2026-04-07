#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간소화된 폴더 구조 생성 스크립트
기존의 복잡한 departments/job_roles/positions/employees 구조를 
간단한 부서/직원 구조로 변경
"""

import os
import shutil
from pathlib import Path
import json
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplifiedStructureCreator:
    """간소화된 폴더 구조 생성기"""
    
    def __init__(self, base_dir="results"):
        self.base_dir = Path(base_dir)
        self.old_structure_backup = Path(f"{base_dir}_backup_{self._get_timestamp()}")
        
    def _get_timestamp(self):
        """타임스탬프 생성"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def backup_existing_structure(self):
        """기존 구조 백업"""
        if self.base_dir.exists():
            logger.info(f"기존 구조를 {self.old_structure_backup}로 백업 중...")
            shutil.copytree(self.base_dir, self.old_structure_backup)
            logger.info("백업 완료")
        else:
            logger.info("기존 구조가 없어 백업을 건너뜁니다.")
    
    def create_simplified_structure(self):
        """간소화된 구조 생성"""
        logger.info("간소화된 폴더 구조 생성 중...")
        
        # 기본 디렉토리 생성
        base_dirs = [
            "global_reports",      # 전체 종합 보고서
            "models",             # 저장된 모델들
            "temp",               # 임시 파일들
            "exports"             # 내보내기 파일들
        ]
        
        for dir_name in base_dirs:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"생성됨: {dir_path}")
        
        # 샘플 부서 구조 생성 (예시)
        sample_departments = [
            "Human_Resources",
            "Information_Technology", 
            "Sales",
            "Marketing",
            "Finance",
            "Operations"
        ]
        
        for dept in sample_departments:
            dept_path = self.base_dir / dept
            dept_path.mkdir(parents=True, exist_ok=True)
            
            # 각 부서에 샘플 직원 폴더 생성 (예시)
            for i in range(1, 4):  # 각 부서당 3명의 샘플 직원
                employee_id = f"employee_{dept[:2].lower()}{i:03d}"
                employee_path = dept_path / employee_id
                employee_path.mkdir(parents=True, exist_ok=True)
                
                # 하위 폴더들
                subfolders = ["visualizations", "reports", "analysis"]
                for subfolder in subfolders:
                    (employee_path / subfolder).mkdir(parents=True, exist_ok=True)
                
                # 샘플 정보 파일 생성
                employee_info = {
                    "employee_id": employee_id,
                    "department": dept,
                    "created_at": self._get_timestamp(),
                    "structure_version": "simplified_v1.0"
                }
                
                with open(employee_path / "employee_info.json", 'w', encoding='utf-8') as f:
                    json.dump(employee_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"부서 구조 생성 완료: {dept}")
        
        logger.info("간소화된 구조 생성 완료!")
    
    def migrate_existing_data(self):
        """기존 데이터를 새 구조로 마이그레이션"""
        if not self.old_structure_backup.exists():
            logger.info("백업된 데이터가 없어 마이그레이션을 건너뜁니다.")
            return
        
        logger.info("기존 데이터 마이그레이션 시작...")
        
        # 기존 departments 폴더에서 데이터 찾기
        old_departments_path = self.old_structure_backup / "departments"
        if not old_departments_path.exists():
            logger.info("기존 departments 폴더가 없습니다.")
            return
        
        migrated_count = 0
        
        # 기존 복잡한 구조에서 데이터 추출
        for dept_folder in old_departments_path.iterdir():
            if not dept_folder.is_dir():
                continue
                
            dept_name = dept_folder.name
            new_dept_path = self.base_dir / dept_name
            new_dept_path.mkdir(parents=True, exist_ok=True)
            
            # job_roles/positions/employees 구조에서 직원 데이터 찾기
            self._extract_employee_data_recursive(dept_folder, new_dept_path, dept_name)
            migrated_count += 1
        
        logger.info(f"마이그레이션 완료: {migrated_count}개 부서 처리됨")
    
    def _extract_employee_data_recursive(self, source_path, target_dept_path, dept_name):
        """재귀적으로 직원 데이터 추출"""
        for item in source_path.rglob("employee_*"):
            if item.is_dir():
                employee_id = item.name
                new_employee_path = target_dept_path / employee_id
                
                if new_employee_path.exists():
                    logger.warning(f"중복된 직원 ID: {employee_id}, 건너뜀")
                    continue
                
                try:
                    # 직원 폴더 전체 복사
                    shutil.copytree(item, new_employee_path)
                    
                    # 메타데이터 업데이트
                    self._update_employee_metadata(new_employee_path, dept_name)
                    
                    logger.info(f"마이그레이션됨: {dept_name}/{employee_id}")
                except Exception as e:
                    logger.error(f"마이그레이션 실패 {employee_id}: {e}")
    
    def _update_employee_metadata(self, employee_path, dept_name):
        """직원 메타데이터 업데이트"""
        info_file = employee_path / "employee_info.json"
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                # 새 구조 정보 추가
                info["migrated_at"] = self._get_timestamp()
                info["structure_version"] = "simplified_v1.0"
                info["department"] = dept_name
                
                with open(info_file, 'w', encoding='utf-8') as f:
                    json.dump(info, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                logger.error(f"메타데이터 업데이트 실패: {e}")
    
    def generate_structure_report(self):
        """구조 변경 보고서 생성"""
        report = {
            "migration_timestamp": self._get_timestamp(),
            "structure_version": "simplified_v1.0",
            "changes": {
                "removed_levels": ["job_roles", "positions"],
                "new_structure": "results/부서명/employee_ID/",
                "old_structure": "results/departments/부서명/job_roles/직무명/positions/직급명/employees/employee_ID/"
            },
            "departments": [],
            "total_employees": 0
        }
        
        # 현재 구조 스캔
        for dept_folder in self.base_dir.iterdir():
            if dept_folder.is_dir() and not dept_folder.name.startswith('.') and dept_folder.name not in ['global_reports', 'models', 'temp', 'exports']:
                employees = list(dept_folder.glob("employee_*"))
                report["departments"].append({
                    "name": dept_folder.name,
                    "employee_count": len(employees)
                })
                report["total_employees"] += len(employees)
        
        # 보고서 저장
        report_path = self.base_dir / "global_reports" / f"structure_migration_report_{self._get_timestamp()}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"구조 변경 보고서 생성: {report_path}")
        return report
    
    def cleanup_old_backup(self, keep_backup=True):
        """백업 정리"""
        if not keep_backup and self.old_structure_backup.exists():
            logger.info(f"백업 폴더 삭제: {self.old_structure_backup}")
            shutil.rmtree(self.old_structure_backup)
        else:
            logger.info(f"백업 폴더 유지: {self.old_structure_backup}")

def main():
    """메인 실행 함수"""
    print("🔧 간소화된 폴더 구조 생성 도구")
    print("=" * 50)
    
    creator = SimplifiedStructureCreator()
    
    try:
        # 1. 기존 구조 백업
        creator.backup_existing_structure()
        
        # 2. 간소화된 구조 생성
        creator.create_simplified_structure()
        
        # 3. 기존 데이터 마이그레이션
        creator.migrate_existing_data()
        
        # 4. 보고서 생성
        report = creator.generate_structure_report()
        
        print("\n✅ 구조 변경 완료!")
        print(f"📊 총 {report['total_employees']}명의 직원 데이터 처리")
        print(f"🏢 {len(report['departments'])}개 부서 구성")
        print("\n새로운 구조:")
        print("results/")
        print("├── 부서명/")
        print("│   └── employee_ID/")
        print("│       ├── visualizations/")
        print("│       ├── reports/")
        print("│       └── analysis/")
        print("├── global_reports/")
        print("├── models/")
        print("├── temp/")
        print("└── exports/")
        
    except Exception as e:
        logger.error(f"구조 생성 실패: {e}")
        print(f"\n❌ 오류 발생: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
