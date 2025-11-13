import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from PyQt5.QtCore import QEventLoop
from PyQt5.QtWidgets import QApplication
from PyQt5.QAxContainer import QAxWidget


class KiwoomAPI(QAxWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

        self.OnEventConnect.connect(self._on_login)
        self.OnReceiveTrData.connect(self._on_tr_data)

        self.login_event_loop: Optional[QEventLoop] = None
        self.tr_event_loop: Optional[QEventLoop] = None

        self._last_tr_data: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Login handling
    # ------------------------------------------------------------------
    def comm_connect(self) -> None:
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def _on_login(self, err_code: int) -> None:
        if err_code != 0:
            print(f"[ERROR] Login failed: {err_code}", file=sys.stderr)

        if self.login_event_loop is not None:
            self.login_event_loop.exit()

    # ------------------------------------------------------------------
    # TR helpers
    # ------------------------------------------------------------------
    def set_input_value(self, key: str, value: str) -> None:
        self.dynamicCall("SetInputValue(QString, QString)", key, value)

    def comm_rq_data(self, rqname: str, trcode: str, prev_next: str, screen_no: str = "1000") -> None:
        self.tr_event_loop = QEventLoop()
        self.dynamicCall(
            "CommRqData(QString, QString, int, QString)",
            rqname,
            trcode,
            int(prev_next),
            screen_no,
        )
        self.tr_event_loop.exec_()

    def _on_tr_data(
        self,
        screen_no: str,
        rqname: str,
        trcode: str,
        record_name: str,
        prev_next: str,
        _1,
        _2,
        _3,
        _4,
    ) -> None:
        data: List[Dict[str, str]] = []
        count = self.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)

        for i in range(count):
            item = {
                "date": self._comm_data(trcode, rqname, i, "일자"),
                "open": self._comm_data(trcode, rqname, i, "시가"),
                "high": self._comm_data(trcode, rqname, i, "고가"),
                "low": self._comm_data(trcode, rqname, i, "저가"),
                "close": self._comm_data(trcode, rqname, i, "현재가"),
                "volume": self._comm_data(trcode, rqname, i, "거래량"),
                "trade_value": self._comm_data(trcode, rqname, i, "거래대금"),
            }
            data.append(item)

        self._last_tr_data[rqname] = {
            "data": data,
            "prev_next": prev_next.strip(),
            "trcode": trcode,
            "record_name": record_name,
        }

        if self.tr_event_loop is not None:
            self.tr_event_loop.exit()

    def _comm_data(self, trcode: str, rqname: str, index: int, item: str) -> str:
        value = self.dynamicCall(
            "GetCommData(QString, QString, int, QString)",
            trcode,
            rqname,
            index,
            item,
        )
        return value.strip()

    # ------------------------------------------------------------------
    # Data retrieval wrappers
    # ------------------------------------------------------------------
    def fetch_daily_ohlcv(self, stock_code: str, base_date: str = "", modified_price: str = "1") -> Dict:
        self.set_input_value("종목코드", stock_code)
        self.set_input_value("기준일자", base_date)
        self.set_input_value("수정주가구분", modified_price)
        self.comm_rq_data("opt10081_req", "opt10081", "0")
        return self._last_tr_data["opt10081_req"]

    def fetch_more_daily(self, stock_code: str) -> Optional[Dict]:
        cached = self._last_tr_data.get("opt10081_req")
        if cached is None:
            return None

        prev_next = cached.get("prev_next", "0")
        if prev_next != "2":
            return None

        self.comm_rq_data("opt10081_req", "opt10081", prev_next)
        return self._last_tr_data["opt10081_req"]

    def get_code_list(self, market: str) -> List[str]:
        codes = self.dynamicCall("GetCodeListByMarket(QString)", market)
        if not codes:
            return []
        return [code for code in codes.split(";") if code]


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def fetch_three_year_daily(kiwoom: KiwoomAPI, stock_code: str, cutoff: str) -> List[Dict[str, str]]:
    combined: List[Dict[str, str]] = []

    response = kiwoom.fetch_daily_ohlcv(stock_code)
    combined.extend(response["data"])

    while response.get("prev_next") == "2":
        time.sleep(0.25)  # TR rate limiting
        response = kiwoom.fetch_more_daily(stock_code)
        if response is None:
            break
        combined.extend(response["data"])
        if combined and combined[-1]["date"] <= cutoff:
            break

    filtered = [row for row in combined if row["date"] >= cutoff]
    return filtered


def main(market_targets: Optional[List[str]] = None) -> None:
    app = QApplication(sys.argv)
    kiwoom = KiwoomAPI()
    kiwoom.comm_connect()

    targets = market_targets or ["0", "10"]  # 0: KOSPI, 10: KOSDAQ
    cutoff_date = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y%m%d")

    output_root = os.path.join(os.getcwd(), "export_daily")
    ensure_directory(output_root)

    for market in targets:
        codes = kiwoom.get_code_list(market)
        market_path = os.path.join(output_root, market)
        ensure_directory(market_path)

        for idx, code in enumerate(codes, start=1):
            try:
                ohlcv = fetch_three_year_daily(kiwoom, code, cutoff_date)

                payload = {
                    "retrieved_at": datetime.now().strftime("%Y%m%d-%H%M%S"),
                    "source_tr": "opt10081",
                    "market": market,
                    "stock_code": code,
                    "cutoff_date": cutoff_date,
                    "records": ohlcv,
                }

                save_json(os.path.join(market_path, f"{code}.json"), payload)
                print(f"[{market}] {idx}/{len(codes)} {code} -> {len(ohlcv)} rows")

                time.sleep(0.25)  # 추가 TR 제한 완충
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] {code}: {exc}", file=sys.stderr)
                time.sleep(1.0)

    app.exit()
    sys.exit(0)


if __name__ == "__main__":
    main()

