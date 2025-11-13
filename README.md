## Overview
- `kiwoom_batch_downloader.py`: Fetches three years of OHLCV data for all KOSPI (`0`) and KOSDAQ (`10`) codes via Kiwoom OpenAPI+ `opt10081`.
- `kiwoom_condition_trader.py`: 조건검색 실시간 매매, 자금 분할, 위험 필터, 트레일링 손익 관리, 적응형 피드백, 차트/로그 기록 포함 자동매매 엔진.
- `config/trader_config.example.json`: 계좌·조건식·예산·위험제어 옵션을 외부화한 JSON 템플릿.

## Prerequisites
- Windows 환경에서 영웅문 HTS / 모의투자 계좌 로그인 유지.
- Kiwoom OpenAPI+ 모듈 설치 및 인증 (키움증권 OpenAPI+ 개발가이드, 2023-09-18 배포본).
- Python 3.9 이상.
- 필수 패키지: `PyQt5`
- 선택 패키지: `matplotlib` (체결 차트 저장 기능 사용 시)

```
pip install PyQt5
# 차트 저장까지 활용하려면
pip install matplotlib
```

## Data Downloader Usage
1. 영웅문에서 OpenAPI+ 로그인 상태 유지.
2. PowerShell 또는 CMD에서 실행:
   ```
   python kiwoom_batch_downloader.py
   ```
3. 결과: `export_daily/<시장코드>/<종목코드>.json`
   - `retrieved_at`, `source_tr`, `cutoff_date` 메타데이터 포함.
   - 데이터는 OpenAPI+ `opt10081` (주식일봉차트) 응답 그대로 저장.

## Condition Trader Usage
1. 예시 설정 복사:
   ```
   copy config\trader_config.example.json config\trader_config.json
   ```
2. `config/trader_config.json` 수정
   - `account_no`: 모의투자 계좌번호
   - `condition_index`, `condition_name`: 영웅문 조건검색에서 확인
   - `total_budget`: 총투입자금(원)
   - `max_positions`: 동시 보유 종목수
   - `per_symbol_budget`: 특정 종목에 대한 개별 예산(선택)
   - `blocked_minutes`: 동일 종목 재진입 금지 시간(분)
   - `trading_start` / `trading_end`: 매매 허용 구간 (예: 09:20~15:15)
   - `profit_take_pct`, `stop_loss_pct`, `trailing_drop_pct`: 기본 익절/손절/트레일링 비율
   - `timeout_minutes`: 보유 기간 초과 시 강제 청산(분)
   - `banned_keywords`, `banned_states`: 관리·ETF·ETN·우선주 등 제외 규칙
   - `chart_minutes`: 체결 후 저장할 분봉 데이터 길이
   - `volatility_lookback`, `volatility_buffer`: 전일 변동성 기반 동적 손익 조정
   - `adaptive_enabled`: 피드백 리포트 생성 여부
3. 실행:
   ```
   python kiwoom_condition_trader.py --config config\trader_config.json
   ```
4. 동작
   - `SendCondition` 실시간 모드로 조건검색 결과 감시.
   - 거래 시간(기본 09:20~15:15) 외, 위험·ETF·ETN·우선주·거래정지 종목은 진입 차단.
   - 최우선 매도/매수호가를 읽어 중간가(limit) 주문 → 3초 내 미체결 시 자동 시장가 전환.
   - 호가 잔량 부족 시 분할 주문, 체결 누적량을 기준으로 포지션 생성.
   - 익절 1% / 손절 3% 기본값에 전일 변동성 계수를 반영해 동적으로 조정.
   - 익절 구간 진입 후 고점 대비 1% 하락 시 트레일링 청산, 5분 경과 시 타임아웃 청산.
   - 청산 시 손익·사유 로그, `logs/trade_log.csv`, `logs/feedback.json` 업데이트 및 `charts/`에 분봉 차트 저장(선택).

## Risk & Feedback Modules
- **재진입 제한**: 동일 종목은 청산 후 `blocked_minutes` 동안 매수 금지.
- **전일 변동성 기반 손익 조절**: 최근 `volatility_lookback` 일간 로그수익률 평균으로 익절/손절 폭 자동 조정.
- **트레일링 익절**: 익절 기준 도달 후 고점 대비 `trailing_drop_pct` 하락 시 자동청산.
- **타임아웃**: 포지션 보유 시간이 `timeout_minutes` 초과하면 강제 청산.
- **체결 로그/차트**: `logs/` 디렉터리에 거래 로그, `charts/` 디렉터리에 분봉 스냅샷 저장(옵션).
- **Adaptive Learning**: 거래 결과 집계 후 `feedback.json`에 조건식 튜닝 시사점 기록(예: 필터 강도 조정, 손절 폭 조절).

## Request Tracker
- (완료) 3년치 KOSPI/KOSDAQ 데이터 일괄 수집 스크립트
- (완료) 모의투자 조건검색 연동 자동매매 환경 구축
- (완료) 외부 JSON 설정 기반 실행 파이프라인
- (완료) 실행 방법 및 참고자료 문서화
- (완료) 재진입 금지, 동적 익절/손절, 리스크 필터, 분할 체결, 로그/차트/피드백 기능 추가

## References
- 키움증권, 「OpenAPI+ 개발가이드 TR 목록 및 레퍼런스」, 2023-09-18 배포본. (https://new.kiwoom.com/ 로그인 후 OpenAPI+ 가이드센터)
- 키움증권, 「모의투자·OpenAPI+ 이용 약관」 (영웅문 내 공지사항 참조, 최신본 확인 필요)

## Notes
- 본 문서는 참고용 일반 정보이며, 자동매매 결과와 법적 책임은 사용자 본인에게 있습니다.
- 실거래 전 모의투자 환경에서 충분히 검증하고, 필요 시 전문가 상담을 받으십시오.
