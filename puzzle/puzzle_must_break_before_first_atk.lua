-- Auto-generated from puzzles.json: must_break_before_first_atk
-- LP/제약/힌트를 포함한 퍼즐 템플릿입니다. (카드코드는 아래 매핑을 변경해 커스터마이즈 가능)
Debug.SetAIName("RL-PUZZLE")
Debug.ReloadFieldBegin(DUEL_ATTACK_FIRST_TURN+DUEL_SIMPLE_AI,4)
Debug.SetPlayerInfo(0,8000,0,0)  -- player
Debug.SetPlayerInfo(1,1600,0,0) -- opponent
Debug.ShowHint("- 먼저 차단 해제 후 공격하세요.\n- 마지막 액션은 ATK여야 합니다.\n- NOTE: RL-Demo")

-- 손패 구성 (RL action 인덱스와 동일한 순서)
Debug.AddCard(40619825, 0, 0, LOCATION_HAND, 0, POS_FACEDOWN)
Debug.AddCard(5318639, 0, 0, LOCATION_HAND, 1, POS_FACEDOWN)
Debug.AddCard(89631139, 0, 0, LOCATION_HAND, 2, POS_FACEDOWN)
Debug.AddCard(89631139, 0, 0, LOCATION_HAND, 3, POS_FACEDOWN)

-- 상대의 '차단' 상황을 상징 (예: 함정 세트)
Debug.AddCard(44095762,1,1,LOCATION_SZONE,2,POS_FACEDOWN) -- Mirror Force
Debug.ReloadFieldEnd()
aux.BeginPuzzle()
