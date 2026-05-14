import os
import json

class PresetManager:
    def __init__(self):
        # Code 폴더(현재 파일 위치)의 부모 폴더 내 presets 폴더를 타겟으로 함
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.preset_dir = os.path.join(self.base_dir, "presets")
        
        # 폴더가 없으면 자동 생성
        if not os.path.exists(self.preset_dir):
            os.makedirs(self.preset_dir)

    def get_preset_list(self):
        """저장된 모든 .json 프리셋 파일 이름 목록을 반환합니다."""
        try:
            if not os.path.exists(self.preset_dir):
                return []
            files = [f for f in os.listdir(self.preset_dir) if f.endswith(".json")]
            return sorted(files)
        except Exception as e:
            print(f"프리셋 목록 로드 중 오류: {e}")
            return []

    def save_preset(self, name, data):
        """분석된 스타일 데이터를 지정된 이름의 JSON 파일로 저장합니다."""
        try:
            if not name.endswith(".json"):
                name += ".json"
            
            path = os.path.join(self.preset_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"프리셋 저장 완료: {path}")
            print(f"========================================")
            return True
        except Exception as e:
            print(f"프리셋 저장 실패: {e}")
            return False

    def load_preset(self, name):
        """선택된 프리셋 파일을 읽어 딕셔너리로 반환합니다."""
        try:
            path = os.path.join(self.preset_dir, name)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"프리셋 로드 실패: {e}")
            return None