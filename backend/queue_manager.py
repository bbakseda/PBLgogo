import time
import random
import threading

class GlobalQueueManager:
    # 🚨 Streamlit의 캐시 초기화 및 인스턴스 재성성 시에도 데이터를 100% 영구 보존하는 클래스 레벨 전역 메모리락
    _global_lock = threading.Lock()
    _global_queue = []
    _global_user_names = {}
    _global_last_heartbeat = {}
    _global_active_user = None

    def __init__(self):
        self.lock = self.__class__._global_lock
        self.queue = self.__class__._global_queue
        self.user_names = self.__class__._global_user_names
        self.last_heartbeat = self.__class__._global_last_heartbeat
        
        # 한국의 귀여운 천연기념물 및 야생동물 이름 닉네임 리스트
        self.available_names = [
            '가마우지', '직박구리', '까막딱따구리', '수리부엉이', '하늘다람쥐', 
            '크낙새', '꾀꼬리', '참새', '고라니', '너구리', '수달', '삵', 
            '반달가슴곰', '족제비', '두루미', '따오기', '저어새', '황조롱이', 
            '쇠제비갈매기', '원앙', '산양', '멧토끼', '다람쥐', '고슴도치', 
            '두더지', '청개구리', '금개구리', '맹꽁이', '까치', '물새', '멧돼지'
        ]

    @property
    def active_user(self):
        return self.__class__._global_active_user

    @active_user.setter
    def active_user(self, value):
        self.__class__._global_active_user = value

    def register_user(self, session_id):
        """세션 ID가 처음 접속했을 때 닉네임을 부여하고 대기열에 등록합니다."""
        with self.lock:
            # 유효 기한 만료 유저 정리
            self._clean_expired_users_no_lock()
            
            if session_id not in self.queue:
                self.queue.append(session_id)
                
            if session_id not in self.user_names:
                # 사용 가능한 닉네임 중 현재 할당되지 않은 이름을 무작위로 추출
                currently_used = set(self.user_names.values())
                pool = [n for n in self.available_names if n not in currently_used]
                if not pool:
                    pool = self.available_names  # 이름이 고갈된 경우 중복 허용
                
                allocated_name = random.choice(pool)
                self.user_names[session_id] = allocated_name
                
            self.last_heartbeat[session_id] = time.time()
            self._update_active_user_no_lock()
            
            return self.user_names[session_id]

    def keep_alive(self, session_id):
        """세션의 생존 신호(하트비트)를 갱신하고 만료된 유저를 자동으로 청소합니다."""
        with self.lock:
            self.last_heartbeat[session_id] = time.time()
            self._clean_expired_users_no_lock()
            self._update_active_user_no_lock()

    def get_queue_status(self, session_id):
        """현재 내 세션 기준 대기열 상태 딕셔너리를 반환합니다."""
        with self.lock:
            self._clean_expired_users_no_lock()
            self._update_active_user_no_lock()
            
            my_name = self.user_names.get(session_id, "미등록 동물")
            
            # 대기열 닉네임 리스트 추출
            queue_names = [self.user_names.get(sid, "알수없는 동물") for sid in self.queue]
            
            # 내 대기 순번 파악 (1부터 시작)
            my_index = -1
            if session_id in self.queue:
                my_index = self.queue.index(session_id) + 1
                
            active_name = self.user_names.get(self.active_user, "없음") if self.active_user else "없음"
            
            return {
                "my_name": my_name,
                "my_turn": my_index,
                "total_waiting": len(self.queue),
                "active_user_name": active_name,
                "queue_list": queue_names,
                "is_my_turn": (self.active_user == session_id)
            }

    def release_turn(self, session_id):
        """작업이 완료되어 다음 차례의 대기자에게 순서를 안전하게 인계합니다."""
        with self.lock:
            if session_id in self.queue:
                self.queue.remove(session_id)
            if session_id in self.user_names:
                del self.user_names[session_id]
            if session_id in self.last_heartbeat:
                del self.last_heartbeat[session_id]
            if self.active_user == session_id:
                self.active_user = None
                
            self._update_active_user_no_lock()

    def _update_active_user_no_lock(self):
        """[내부 함수] 현재 활성 연산자가 없거나 만료된 경우 대기열 맨 앞사람으로 교체합니다."""
        if not self.queue:
            self.active_user = None
            return

        # 활성 사용자가 지정되지 않았거나, 대기열 목록에서 이탈한 경우 맨 앞사람으로 권한 양도
        if self.active_user not in self.queue:
            self.active_user = self.queue[0]

    def _clean_expired_users_no_lock(self, timeout_seconds=30):
        """[내부 함수] 30초 동안 하트비트 응답이 없는 잠수 접속자를 대기열에서 퇴출합니다."""
        now = time.time()
        expired_sessions = []
        
        for sid, last_time in self.last_heartbeat.items():
            if now - last_time > timeout_seconds:
                expired_sessions.append(sid)
                
        for sid in expired_sessions:
            if sid in self.queue:
                self.queue.remove(sid)
            if sid in self.user_names:
                del self.user_names[sid]
            if sid in self.last_heartbeat:
                del self.last_heartbeat[sid]
            if self.active_user == sid:
                self.active_user = None
