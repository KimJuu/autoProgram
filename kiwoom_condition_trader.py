"""Kiwoom OpenAPI+ condition-based automated trading utilities.

This module wires up condition search feeds, order management, and risk
controls to provide a paper-trading environment for 전략 실험. The core class,
``ConditionAutoTrader``, coordinates login, TR requests, 실시간 이벤트, 분할 체결,
트레일링 손익 관리, 로그/차트 기록, 그리고 간단한 피드백 루프까지 수행한다.

구성은 JSON 파일을 통해 외부에서 조정하며, 각 기능은 키움 OpenAPI+ 공식
문서(2023-09-18 배포본)를 기반으로 구현되었다.
"""

import csv
import importlib
import json
import logging
import math
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QAxContainer import QAxWidget


@dataclass
class OrderConfig:
    """Configuration values controlling account info, budgets, and risk rules."""

    total_budget: int = 40_000_000  # KRW
    max_positions: int = 4
    condition_name: str = ""
    condition_index: int = -1
    account_no: str = ""
    screen_no: str = "2000"
    price_unit: int = 1  # Board lot size (1 share for KRX equities)
    per_symbol_budget: Dict[str, int] = field(default_factory=dict)
    blocked_minutes: int = 60
    trading_start: str = "09:20"
    trading_end: str = "15:15"
    profit_take_pct: float = 0.01
    stop_loss_pct: float = 0.03
    trailing_drop_pct: float = 0.01
    timeout_minutes: int = 5
    banned_keywords: List[str] = field(
        default_factory=lambda: ["우", "ETF", "ETN", "스팩", "관리", "선물", "인버스"]
    )
    banned_states: Set[str] = field(default_factory=lambda: {"관리종목", "거래정지", "투자주의"})
    chart_minutes: int = 240
    volatility_lookback: int = 20
    volatility_buffer: float = 0.5
    adaptive_enabled: bool = True
    log_dir: str = "logs"
    chart_dir: str = "charts"


@dataclass
class PositionInfo:
    """Runtime snapshot for a single open position."""

    code: str
    entry_price: float
    quantity: int
    entry_time: datetime
    high_price: float
    stop_loss: float
    take_profit: float
    volatility_factor: float
    timeout_deadline: datetime
    reached_take_profit: bool = False

    def update_high(self, price: float) -> None:
        if price > self.high_price:
            self.high_price = price

    def trailing_stop_triggered(self, price: float, trailing_pct: float) -> bool:
        if self.high_price <= 0:
            return False
        drop = (self.high_price - price) / self.high_price
        return drop >= trailing_pct and price > self.entry_price


@dataclass
class OrderStatus:
    """Tracks progress of an outstanding limit/market order."""

    code: str
    quantity: int
    placed_at: datetime
    order_type: str  # "buy" or "sell"
    order_no: Optional[str] = None
    filled: int = 0
    price: float = 0.0
    executed_value: float = 0.0

    def remaining(self) -> int:
        return max(self.quantity - self.filled, 0)


class ConditionAutoTrader(QAxWidget):
    """Condition-search driven trading engine built on Kiwoom OpenAPI+.

    The class encapsulates:

    - Condition subscription (initial + real-time) and entry filtering.
    - Order management with mid-price limit placement, automatic market
      conversion, and split sizing when order-book depth is thin.
    - Risk controls such as trading windows, instrument bans, volatility-aware
      stop/target levels, trailing profits, and timeouts.
    - Logging, chart capture, and adaptive feedback to help tune search formulas.

    The behaviour of the engine is configured through :class:`OrderConfig`.

    References:
      - Kiwoom OpenAPI+ Developer Guide, Condition Search TR/Real API (2023-09-18)
      - Kiwoom OpenAPI+ Reference, SendOrder / SendCondition specifications
    """

    def __init__(self, config: OrderConfig) -> None:
        super().__init__()
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

        self.config = config
        self.login_event_loop: Optional[QEventLoop] = None
        self.tr_event_loop: Optional[QEventLoop] = None

        self._last_tr_data: Dict[str, Dict] = {}
        self.positions: Dict[str, PositionInfo] = {}
        self.pending_orders: Dict[str, OrderStatus] = {}
        self.blocked_codes: Dict[str, datetime] = {}
        self.trade_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.trade_history: List[Dict[str, str]] = []
        self.orderbook: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.pending_entries: Dict[str, Dict[str, float]] = {}
        self.pending_exits: Dict[str, Dict[str, float]] = {}
        self.real_registered: Set[str] = set()
        self.exiting_codes: Set[str] = set()

        self._init_directories()
        self._init_logger()

        # Event wiring
        self.OnEventConnect.connect(self._on_login)
        self.OnReceiveTrData.connect(self._on_tr_data)
        self.OnReceiveTrCondition.connect(self._on_tr_condition)
        self.OnReceiveRealCondition.connect(self._on_real_condition)
        self.OnReceiveChejanData.connect(self._on_chejan)
        self.OnReceiveRealData.connect(self._on_real_data)

        self.housekeeping_timer = QTimer()
        self.housekeeping_timer.setInterval(1000)
        self.housekeeping_timer.timeout.connect(self._housekeeping)

    def _init_directories(self) -> None:
        """Ensure log/chart directories exist before runtime files are written."""
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.chart_dir).mkdir(parents=True, exist_ok=True)

    def _init_logger(self) -> None:
        """Configure application-wide logging sinks (file + stdout)."""
        log_path = Path(self.config.log_dir) / "trader.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    # ------------------------------------------------------------------ #
    # Login / basic helpers
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Start the login process and block on the associated event loop."""
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def _on_login(self, err_code: int) -> None:
        """Handle login callback, start timers, and validate account config."""
        if err_code != 0:
            logging.error("Login failed: %s", err_code)
        else:
            accounts = self._get_accounts()
            logging.info("Logged in. Accounts: %s", accounts)
            if self.config.account_no and self.config.account_no not in accounts:
                logging.error("Configured account %s not available.", self.config.account_no)
            self.housekeeping_timer.start()
        if self.login_event_loop is not None:
            self.login_event_loop.exit()

    def _get_accounts(self) -> Set[str]:
        """Return the list of account numbers associated with the session."""
        raw = self.dynamicCall("GetLoginInfo(QString)", "ACCNO")
        return {acc.strip() for acc in raw.split(";") if acc.strip()}

    def set_input_value(self, key: str, value: str) -> None:
        """Delegate to Kiwoom SetInputValue to configure TR parameters."""
        self.dynamicCall("SetInputValue(QString, QString)", key, value)

    def comm_rq_data(self, rqname: str, trcode: str, prev_next: str, screen_no: str) -> None:
        """Send a TR request and block until the response event fires."""
        self.tr_event_loop = QEventLoop()
        self.dynamicCall(
            "CommRqData(QString, QString, int, QString)", rqname, trcode, int(prev_next), screen_no
        )
        self.tr_event_loop.exec_()

    def _on_tr_data(
        self,
        screen_no: str,
        rqname: str,
        trcode: str,
        record_name: str,
        prev_next: str,
        *_,
    ) -> None:
        """Generic TR callback used by helper fetches (daily/minute data)."""
        count = self.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        rows = []
        for i in range(count):
            row = {
                "현재가": self._comm_data(trcode, rqname, i, "현재가"),
                "거래량": self._comm_data(trcode, rqname, i, "거래량"),
                "종목명": self._comm_data(trcode, rqname, i, "종목명"),
            }
            rows.append(row)

        self._last_tr_data[rqname] = {
            "rows": rows,
            "prev_next": prev_next.strip(),
            "trcode": trcode,
            "record_name": record_name,
        }
        if self.tr_event_loop is not None:
            self.tr_event_loop.exit()

    def _comm_data(self, trcode: str, rqname: str, index: int, item: str) -> str:
        """Fetch and strip a field from the most recent TR response cache."""
        value = self.dynamicCall(
            "GetCommData(QString, QString, int, QString)",
            trcode,
            rqname,
            index,
            item,
        )
        return value.strip()

    def _on_real_data(self, code: str, real_type: str, real_data: str) -> None:
        if real_type == "주식호가":
            ask = self.dynamicCall("GetCommRealData(QString, int)", code, 41)
            bid = self.dynamicCall("GetCommRealData(QString, int)", code, 51)
            ask_size = self.dynamicCall("GetCommRealData(QString, int)", code, 61)
            bid_size = self.dynamicCall("GetCommRealData(QString, int)", code, 71)
            try:
                self.orderbook[code]["ask"] = abs(float(ask))
                self.orderbook[code]["bid"] = abs(float(bid))
                self.orderbook[code]["ask_size"] = abs(float(ask_size))
                self.orderbook[code]["bid_size"] = abs(float(bid_size))
            except ValueError:
                pass
        elif real_type == "주식체결":
            price = self.dynamicCall("GetCommRealData(QString, int)", code, 10)
            volume = self.dynamicCall("GetCommRealData(QString, int)", code, 13)
            try:
                self.orderbook[code]["last"] = abs(float(price))
                self.orderbook[code]["last_volume"] = abs(float(volume))
            except ValueError:
                pass

    # ------------------------------------------------------------------ #
    # Condition load & subscription
    # ------------------------------------------------------------------ #
    def load_conditions(self) -> Dict[int, str]:
        ret = self.dynamicCall("GetConditionLoad()")
        if ret != 1:
            raise RuntimeError("Condition load failed.")
        time.sleep(0.5)

        raw = self.dynamicCall("GetConditionNameList()")
        result: Dict[int, str] = {}
        if not raw:
            return result
        for part in raw.split(";"):
            if not part:
                continue
            idx, name = part.split("^")
            result[int(idx)] = name
        logging.info("Loaded %d conditions.", len(result))
        return result

    def activate_condition(self) -> None:
        if self.config.condition_index < 0 or not self.config.condition_name:
            raise ValueError("Condition name/index must be configured.")
        logging.info(
            "Activating condition: %s (%s)", self.config.condition_name, self.config.condition_index
        )
        ret = self.dynamicCall(
            "SendCondition(QString, QString, int, int)",
            self.config.screen_no,
            self.config.condition_name,
            self.config.condition_index,
            1,  # 1: real-time + initial search
        )
        if ret != 1:
            raise RuntimeError("SendCondition failed.")

    def deactivate_condition(self) -> None:
        self.dynamicCall(
            "SendConditionStop(QString, QString, int)",
            self.config.screen_no,
            self.config.condition_name,
            self.config.condition_index,
        )
        logging.info("Condition subscription stopped.")

    # ------------------------------------------------------------------ #
    # Condition callbacks
    # ------------------------------------------------------------------ #
    def _on_tr_condition(self, screen_no: str, code_list: str, condition_name: str, condition_index: int, *_):
        """Process the initial batch of codes returned by SendCondition."""
        codes = [code for code in code_list.split(";") if code]
        logging.info("Initial condition matches (%s): %s", condition_name, codes)
        for code in codes:
            self._attempt_entry(code)

    def _on_real_condition(self, code: str, event_type: str, condition_name: str, condition_index: str) -> None:
        """Handle real-time inclusion/exclusion events from condition search."""
        if event_type == "I":  # Enter
            logging.info("Condition IN: %s", code)
            self._attempt_entry(code)
        elif event_type == "D":  # Exit
            logging.info("Condition OUT: %s", code)

    def _attempt_entry(self, code: str) -> None:
        """Apply gating rules and, if permitted, queue a new long position."""
        now = datetime.now()
        # 장 시작 전/후 등 시간 조건 우선 확인
        if not self._is_within_trading_window(now):
            logging.debug("Outside trading window. Skipping %s", code)
            return
        if len(self.positions) >= self.config.max_positions:
            logging.info("Position limit reached; skipping %s", code)
            return
        # 이미 보유 중이거나 주문이 진행 중인 경우는 제외
        if (
            code in self.positions
            or code in self.pending_entries
            or any(order.code == code for order in self.pending_orders.values())
        ):
            logging.debug("Already holding or pending order for %s", code)
            return
        # 지정된 재진입 금지 시간 또는 위험 관리 대상 여부 확인
        if not self._passes_block(code, now):
            logging.info("Code %s is blocked from re-entry.", code)
            return
        if not self._is_allowed_asset(code):
            logging.info("Filtered out %s due to risk classification.", code)
            return

        last_price = self._request_last_price(code)
        if last_price <= 0:
            logging.warning("Invalid last price for %s", code)
            return

        budget = self.config.per_symbol_budget.get(code, 0)
        if budget <= 0:
            budget = self.config.total_budget // self.config.max_positions

        volatility_factor = self._compute_volatility_factor(code)
        take_profit, stop_loss = self._calculate_risk_levels(last_price, volatility_factor)

        # 매수 수량 = (예산 / 가격)을 호가단위에 맞춰 보정
        quantity = max(int(budget // last_price), 0)
        quantity -= quantity % self.config.price_unit

        if quantity <= 0:
            logging.warning("Computed quantity zero for %s", code)
            return

        # 체결을 추적하기 위해 임시 버퍼에 목표 수량과 실행 금액을 저장
        self.pending_entries[code] = {
            "target_qty": quantity,
            "filled_qty": 0,
            "executed_value": 0.0,
            "volatility_factor": volatility_factor,
        }
        logging.info(
            "Attempting entry %s qty=%s price≈%s TP=%.2f SL=%.2f (vol=%.3f)",
            code,
            quantity,
            last_price,
            take_profit,
            stop_loss,
            volatility_factor,
        )
        self._place_entry_order(code, quantity, last_price)

    def _is_within_trading_window(self, now: datetime) -> bool:
        """Check whether the current timestamp lies within the trading window."""
        start_hour, start_minute = map(int, self.config.trading_start.split(":"))
        end_hour, end_minute = map(int, self.config.trading_end.split(":"))
        start = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        end = now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
        return start <= now <= end

    def _passes_block(self, code: str, now: datetime) -> bool:
        """True if the symbol is not currently in the re-entry cooldown window."""
        expiry = self.blocked_codes.get(code)
        if expiry is None:
            return True
        if now >= expiry:
            self.blocked_codes.pop(code, None)
            return True
        return False

    def _is_allowed_asset(self, code: str) -> bool:
        """Return False for 관리·ETF·ETN·우선주·거래정지 등 위험 분류 종목."""
        name = self.dynamicCall("GetMasterCodeName(QString)", code)
        upper_name = name.upper() if name else ""
        if any(keyword.upper() in upper_name for keyword in self.config.banned_keywords):
            return False
        state = self.dynamicCall("GetMasterStockState(QString)", code)
        if state:
            for banned in self.config.banned_states:
                if banned in state:
                    return False
        return True

    def _compute_volatility_factor(self, code: str) -> float:
        try:
            rows = self._fetch_daily_series(code, self.config.volatility_lookback)
            closes: List[float] = []
            for row in rows:
                try:
                    value = abs(float(row["close"]))
                except (ValueError, TypeError):
                    continue
                if value > 0:
                    closes.append(value)
            if len(closes) < 2:
                return 1.0
            returns = [
                abs(math.log(closes[idx] / closes[idx - 1]))
                for idx in range(1, len(closes))
                if closes[idx - 1] > 0
            ]
            if not returns:
                return 1.0
            avg = sum(returns) / len(returns)
            return max(avg / self.config.volatility_buffer, 0.1)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Volatility fetch failed for %s: %s", code, exc)
            return 1.0

    def _calculate_risk_levels(self, price: float, volatility_factor: float) -> Tuple[float, float]:
        tp_pct = self.config.profit_take_pct * (1 + volatility_factor)
        sl_pct = self.config.stop_loss_pct * max(1 - volatility_factor, 0.2)
        tp_pct = max(tp_pct, 0.005)
        sl_pct = max(sl_pct, 0.005)
        take_profit = price * (1 + tp_pct)
        stop_loss = price * (1 - sl_pct)
        return take_profit, stop_loss

    def _place_entry_order(self, code: str, quantity: int, reference_price: float) -> None:
        """Submit limit orders (possibly split) to start building the position."""
        self._ensure_real_subscription(code)
        asksize = int(self.orderbook[code].get("ask_size", quantity))
        if asksize > 0 and asksize < quantity:
            parts = self._split_quantity(quantity, asksize)
        else:
            parts = [quantity]

        for idx, part in enumerate(parts):
            rqname = f"buy_{code}_{int(time.time())}_{idx}"
            price = self._mid_price(code, reference_price)
            order = OrderStatus(code=code, quantity=part, placed_at=datetime.now(), order_type="buy", price=price)
            self.pending_orders[rqname] = order
            self._send_limit_order(rqname, order)
            self._schedule_market_conversion(rqname)

    def _split_quantity(self, total: int, lot: int) -> List[int]:
        """Return a list of child order sizes based on available depth."""
        if lot <= 0:
            return [total]
        parts = []
        remaining = total
        while remaining > 0:
            chunk = min(lot, remaining)
            parts.append(chunk)
            remaining -= chunk
        return parts

    def _ensure_real_subscription(self, code: str) -> None:
        """Register real-time feeds (체결/호가) for the specified ticker once."""
        if code in self.real_registered:
            return
        fid_list = "10;13;41;51;61;71"
        self.dynamicCall("SetRealReg(QString, QString, QString, QString)", self.config.screen_no, code, fid_list, "0")
        self.real_registered.add(code)

    def _mid_price(self, code: str, fallback: float) -> float:
        """Calculate a fair mid-price using latest bid/ask with fallback."""
        ask = self.orderbook[code].get("ask")
        bid = self.orderbook[code].get("bid")
        if ask and bid:
            return self._adjust_price((ask + bid) / 2)
        return self._adjust_price(fallback)

    def _adjust_price(self, price: float) -> float:
        """Convert a raw price to the nearest legal tick size."""
        if price <= 0:
            return 0.0
        if price < 1000:
            tick = 1
        elif price < 5000:
            tick = 5
        elif price < 10000:
            tick = 10
        elif price < 50000:
            tick = 50
        elif price < 100000:
            tick = 100
        else:
            tick = 500
        return round(price / tick) * tick

    def _send_limit_order(self, rqname: str, order: OrderStatus) -> None:
        """Submit a limit order (buy/sell) via SendOrder."""
        price = int(order.price)
        if price <= 0:
            price = 0
        ret = self.dynamicCall(
            "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
            rqname,
            self.config.screen_no,
            self.config.account_no,
            1 if order.order_type == "buy" else 2,
            order.code,
            order.quantity,
            price,
            "00",  # Limit order
            "",
        )
        if ret != 0:
            logging.error("SendOrder limit failed (%s) -> %s", order.code, ret)
            self.pending_orders.pop(rqname, None)
        else:
            logging.info("Limit order submitted %s %s qty=%s price=%s", order.order_type, order.code, order.quantity, price)

    def _schedule_market_conversion(self, rqname: str) -> None:
        """Fallback to market execution if the limit has not filled."""
        QTimer.singleShot(3000, lambda: self._convert_to_market(rqname))

    def _convert_to_market(self, rqname: str) -> None:
        """Cancel the resting limit order and submit the remainder at market."""
        order = self.pending_orders.get(rqname)
        if order is None:
            return
        if order.remaining() <= 0:
            return
        if order.order_no is None:
            logging.debug("Awaiting order number for %s", rqname)
            QTimer.singleShot(1000, lambda: self._convert_to_market(rqname))
            return
        logging.info("Converting to market order for %s", order.code)
        cancel_ret = self.dynamicCall(
            "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
            f"cancel_{rqname}",
            self.config.screen_no,
            self.config.account_no,
            3 if order.order_type == "buy" else 4,
            order.code,
            0,
            0,
            "00",
            order.order_no,
        )
        if cancel_ret != 0:
            logging.warning("Cancel attempt failed for %s -> %s", order.code, cancel_ret)
        market_ret = self.dynamicCall(
            "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
            f"market_{rqname}",
            self.config.screen_no,
            self.config.account_no,
            1 if order.order_type == "buy" else 2,
            order.code,
            order.remaining(),
            0,
            "03",  # Market
            "",
        )
        if market_ret != 0:
            logging.error("Market conversion failed for %s -> %s", order.code, market_ret)
        else:
            logging.info("Market order submitted for remaining qty=%s of %s", order.remaining(), order.code)

    def _housekeeping(self) -> None:
        """Periodic maintenance: release cooldowns, evaluate exits, prune orders."""
        now = datetime.now()
        self._cleanup_blocks(now)
        self._evaluate_positions(now)
        self._cleanup_orders(now)

    def _cleanup_blocks(self, now: datetime) -> None:
        """Remove tickers from the blocked list once cooldown expires."""
        expired = [code for code, expiry in self.blocked_codes.items() if expiry <= now]
        for code in expired:
            self.blocked_codes.pop(code, None)

    def _cleanup_orders(self, now: datetime) -> None:
        """Drop completed orders and reset pending-entry buffers if unused."""
        to_remove: List[Tuple[str, str]] = []
        for rqname, order in self.pending_orders.items():
            if order.remaining() <= 0 and (now - order.placed_at).total_seconds() > 5:
                to_remove.append((rqname, order.code))
        for rqname, code in to_remove:
            self.pending_orders.pop(rqname, None)
            entry = self.pending_entries.get(code)
            if entry and entry.get("filled_qty", 0) == 0:
                self.pending_entries.pop(code, None)

    def _evaluate_positions(self, now: datetime) -> None:
        """Check each position against stop-loss, trailing, and timeout logic."""
        for code, position in list(self.positions.items()):
            price = self._current_price(code)
            if price <= 0:
                continue
            # 최신 체결가로 고점을 갱신하면 트레일링 조건 계산에 활용된다.
            position.update_high(price)

            if not position.reached_take_profit and price >= position.take_profit:
                logging.info("Take-profit threshold reached for %s", code)
                position.reached_take_profit = True

            # 절대 손절 조건을 가장 먼저 평가한다.
            if price <= position.stop_loss:
                logging.info("Stop loss triggered for %s", code)
                self._initiate_exit(code, "stop_loss")
                continue

            # 익절 후 고점 대비 하락 폭이 크면 트레일링 손절을 발동한다.
            if position.reached_take_profit and position.trailing_stop_triggered(
                price, self.config.trailing_drop_pct
            ):
                logging.info("Trailing stop triggered for %s", code)
                self._initiate_exit(code, "trailing")
                continue

            # 설정된 보유 시간 초과 시 강제 청산.
            if now >= position.timeout_deadline:
                logging.info("Timeout exit for %s", code)
                self._initiate_exit(code, "timeout")

    def _current_price(self, code: str) -> float:
        """Return the latest known price (real-time cache or master last price)."""
        price = self.orderbook[code].get("last")
        if price:
            return price
        raw = self.dynamicCall("GetMasterLastPrice(QString)", code)
        try:
            return abs(float(raw))
        except ValueError:
            return 0.0

    def _initiate_exit(self, code: str, reason: str) -> None:
        """Prepare a sell order (possibly split) for the active position."""
        if code in self.exiting_codes:
            return
        position = self.positions.get(code)
        if position is None:
            return
        self.exiting_codes.add(code)
        # 현재 호가 정보를 활용해 중간가(limit)를 산정한다.
        price = self._mid_price(code, position.high_price)
        quantity = position.quantity
        self.pending_exits[code] = {
            "target_qty": quantity,
            "filled_qty": 0,
            "executed_value": 0.0,
            "reason": reason,
        }
        # 실시간 호가가 등록되어 있지 않으면 추가 등록
        self._ensure_real_subscription(code)
        bid_size = int(self.orderbook[code].get("bid_size", quantity))
        # 매도 호가 잔량이 부족한 경우 분할 주문으로 나눈다.
        parts = self._split_quantity(quantity, bid_size)
        for idx, part in enumerate(parts):
            rqname = f"sell_{code}_{int(time.time())}_{idx}"
            order = OrderStatus(
                code=code,
                quantity=part,
                placed_at=datetime.now(),
                order_type="sell",
                price=price,
            )
            self.pending_orders[rqname] = order
            self._send_limit_order(rqname, order)
            self._schedule_market_conversion(rqname)

    # ------------------------------------------------------------------ #
    # Order utilities
    # ------------------------------------------------------------------ #
    def _request_last_price(self, code: str) -> int:
        """Return the master last price (used for quick budget calculations)."""
        price_str = self.dynamicCall("GetMasterLastPrice(QString)", code)
        try:
            return abs(int(price_str))
        except ValueError:
            return 0

    def _on_chejan(self, gubun: str, item_cnt: int, fid_list: str) -> None:
        """React to order/position notifications (체결·미체결)."""
        parsed: Dict[str, str] = {}
        for fid in fid_list.split(";"):
            fid = fid.strip()
            if not fid:
                continue
            value = self.dynamicCall("GetChejanData(int)", int(fid)).strip()
            parsed[fid] = value

        code_raw = parsed.get("9001", "")
        code = code_raw[1:] if len(code_raw) > 1 else code_raw
        order_no = parsed.get("9203", "")
        filled_qty = int(parsed.get("911", "0") or 0)
        price = abs(float(parsed.get("302", "0") or 0))

        if not code:
            return

        for rqname, order in list(self.pending_orders.items()):
            if order.code != code:
                continue
            if order.order_no is None and order_no:
                # 주문 번호는 추후 정정/취소를 위해 저장
                order.order_no = order_no
            if filled_qty > 0 and price > 0:
                # 누적 체결 수량·금액을 기반으로 평균 체결가 산출
                order.filled += filled_qty
                order.executed_value += price * filled_qty
                order.price = order.executed_value / order.filled if order.filled else price
                entry_completed = False
                exit_completed = False
                if order.order_type == "buy":
                    entry_completed = self._accumulate_entry(code, filled_qty, price)
                else:
                    exit_completed = self._accumulate_exit(code, filled_qty, price)
                logging.info(
                    "Order update %s %s filled=%s/%s avg_price=%.2f",
                    order.order_type,
                    code,
                    order.filled,
                    order.quantity,
                    order.price,
                )
                if order.remaining() <= 0:
                    self.pending_orders.pop(rqname, None)
                    if order.order_type == "buy" and entry_completed:
                        self._finalize_entry(code)
                    elif order.order_type == "sell" and exit_completed:
                        self._finalize_exit(code)

    def _accumulate_entry(self, code: str, quantity: int, price: float) -> bool:
        """Track partial fills for buy orders; return True when target met."""
        entry = self.pending_entries.get(code)
        if entry is None:
            return False
        entry["filled_qty"] += quantity
        entry["executed_value"] += price * quantity
        return entry["filled_qty"] >= entry["target_qty"]

    def _accumulate_exit(self, code: str, quantity: int, price: float) -> bool:
        """Track partial fills for sell orders; return True when exit complete."""
        exit_info = self.pending_exits.get(code)
        if exit_info is None:
            return False
        exit_info["filled_qty"] += quantity
        exit_info["executed_value"] += price * quantity
        return exit_info["filled_qty"] >= exit_info["target_qty"]

    def _finalize_entry(self, code: str) -> None:
        """Convert completed buy orders into a tracked :class:`PositionInfo`."""
        entry = self.pending_entries.pop(code, None)
        if entry is None:
            return
        quantity = int(entry.get("filled_qty", 0))
        if quantity <= 0:
            return
        executed_value = entry.get("executed_value", 0.0)
        # 체결 금액 / 수량으로 평균 매수가를 계산 (잔여 체결 값이 없는 경우 현재가 사용)
        entry_price = executed_value / quantity if executed_value > 0 else self._current_price(code)
        volatility_factor = entry.get("volatility_factor", 1.0)
        take_profit, stop_loss = self._calculate_risk_levels(entry_price, volatility_factor)
        position = PositionInfo(
            code=code,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            high_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volatility_factor=volatility_factor,
            timeout_deadline=datetime.now() + timedelta(minutes=self.config.timeout_minutes),
        )
        self.positions[code] = position
        self.exiting_codes.discard(code)
        logging.info(
            "Position opened %s qty=%s entry=%.2f TP=%.2f SL=%.2f",
            code,
            quantity,
            entry_price,
            take_profit,
            stop_loss,
        )

    def _finalize_exit(self, code: str) -> None:
        """Wrap up a closed position: calculate PnL, log, and enforce cooldown."""
        exit_info = self.pending_exits.pop(code, None)
        position = self.positions.pop(code, None)
        self.exiting_codes.discard(code)
        if position is None:
            return
        quantity = position.quantity
        if exit_info is not None and exit_info.get("filled_qty"):
            quantity = int(exit_info.get("filled_qty", quantity))
        executed_value = exit_info.get("executed_value", 0.0) if exit_info else 0.0
        # 분할 청산의 평균 청산가를 계산 (없다면 현재가 사용)
        exit_price = executed_value / quantity if executed_value > 0 and quantity > 0 else self._current_price(code)
        pnl = (exit_price - position.entry_price) * position.quantity
        pnl_pct = (exit_price / position.entry_price - 1) if position.entry_price else 0
        reason = exit_info.get("reason", "") if exit_info else ""
        logging.info(
            "Position closed %s qty=%s entry=%.2f exit=%.2f pnl=%.0f (%.3f%%) reason=%s",
            code,
            position.quantity,
            position.entry_price,
            exit_price,
            pnl,
            pnl_pct * 100,
            reason,
        )
        self.blocked_codes[code] = datetime.now() + timedelta(minutes=self.config.blocked_minutes)
        self.trade_history.append(
            {
                "code": code,
                "entry_price": f"{position.entry_price:.2f}",
                "exit_price": f"{exit_price:.2f}",
                "quantity": str(position.quantity),
                "pnl": f"{pnl:.0f}",
                "pnl_pct": f"{pnl_pct*100:.3f}",
                "entry_time": position.entry_time.isoformat(),
                "exit_time": datetime.now().isoformat(),
                "reason": reason,
            }
        )
        self._log_trade(position, exit_price, pnl, pnl_pct)
        self._save_chart(code, position.entry_time)
        if self.config.adaptive_enabled:
            self._update_feedback(position, exit_price, pnl_pct)

    def _fetch_daily_series(self, code: str, count: int) -> List[Dict[str, str]]:
        rqname = f"daily_{code}"
        self.set_input_value("종목코드", code)
        self.set_input_value("기준일자", "")
        self.set_input_value("수정주가구분", "1")
        self.comm_rq_data(rqname, "opt10081", "0", self.config.screen_no)
        rows: List[Dict[str, str]] = []
        available = self.dynamicCall("GetRepeatCnt(QString, QString)", "opt10081", rqname)
        limit = min(available, count)
        for i in range(limit):
            rows.append(
                {
                    "date": self._comm_data("opt10081", rqname, i, "일자"),
                    "high": self._comm_data("opt10081", rqname, i, "고가"),
                    "low": self._comm_data("opt10081", rqname, i, "저가"),
                    "close": self._comm_data("opt10081", rqname, i, "현재가"),
                }
            )
        rows.reverse()
        return rows

    def _fetch_minute_series(self, code: str, count: int) -> List[Dict[str, str]]:
        rqname = f"minute_{code}"
        self.set_input_value("종목코드", code)
        self.set_input_value("틱범위", "1")
        self.set_input_value("수정주가구분", "1")
        self.comm_rq_data(rqname, "opt10080", "0", self.config.screen_no)
        rows: List[Dict[str, str]] = []
        available = self.dynamicCall("GetRepeatCnt(QString, QString)", "opt10080", rqname)
        limit = min(available, count)
        for i in range(limit):
            rows.append(
                {
                    "time": self._comm_data("opt10080", rqname, i, "체결시간"),
                    "close": self._comm_data("opt10080", rqname, i, "현재가"),
                }
            )
        return rows

    def _log_trade(self, position: PositionInfo, exit_price: float, pnl: float, pnl_pct: float) -> None:
        log_path = Path(self.config.log_dir) / "trade_log.csv"
        new_file = not log_path.exists()
        with open(log_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(
                    [
                        "code",
                        "entry_time",
                        "exit_time",
                        "entry_price",
                        "exit_price",
                        "quantity",
                        "pnl",
                        "pnl_pct",
                    ]
                )
            writer.writerow(
                [
                    position.code,
                    position.entry_time.isoformat(),
                    datetime.now().isoformat(),
                    f"{position.entry_price:.2f}",
                    f"{exit_price:.2f}",
                    position.quantity,
                    f"{pnl:.0f}",
                    f"{pnl_pct*100:.3f}",
                ]
            )

    def _save_chart(self, code: str, entry_time: datetime) -> None:
        try:
            plt = importlib.import_module("matplotlib.pyplot")  # type: ignore[assignment]
        except ModuleNotFoundError:
            logging.warning("Matplotlib not installed; skip chart capture.")
            return
        try:
            rows = self._fetch_minute_series(code, self.config.chart_minutes)
            if not rows:
                return
            times: List[str] = []
            prices: List[float] = []
            for row in rows:
                try:
                    price = abs(float(row["close"]))
                except (ValueError, TypeError):
                    continue
                times.append(row["time"])
                prices.append(price)
            if not prices:
                return
            plt.figure(figsize=(10, 4))
            plt.plot(times, prices, label="Close")
            entry_label = entry_time.strftime("%H%M")
            if entry_label in times:
                idx = times.index(entry_label)
                plt.axvline(idx, color="green", linestyle="--", alpha=0.5, label="Entry")
            plt.xticks(rotation=45)
            plt.title(f"{code} trade snapshot")
            plt.tight_layout()
            chart_path = Path(self.config.chart_dir) / f"{code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path)
            plt.close()
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to save chart for %s: %s", code, exc)

    def _update_feedback(self, position: PositionInfo, exit_price: float, pnl_pct: float) -> None:
        stats = self.trade_stats[position.code]
        stats["trades"] += 1
        stats["sum_return"] += pnl_pct
        if pnl_pct > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        overall = self.trade_stats["__portfolio__"]
        overall["trades"] += 1
        overall["sum_return"] += pnl_pct
        if pnl_pct > 0:
            overall["wins"] += 1
        else:
            overall["losses"] += 1
        self._write_feedback_report()

    def _write_feedback_report(self) -> None:
        report = {}
        for code, stats in self.trade_stats.items():
            trades = stats.get("trades", 0)
            if trades == 0:
                continue
            avg_return = stats.get("sum_return", 0) / trades
            win_rate = stats.get("wins", 0) / trades
            suggestion = self._build_suggestion(avg_return, win_rate)
            report[code] = {
                "trades": trades,
                "avg_return": avg_return,
                "win_rate": win_rate,
                "suggestion": suggestion,
            }
        report_path = Path(self.config.log_dir) / "feedback.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

    def _build_suggestion(self, avg_return: float, win_rate: float) -> str:
        if win_rate < 0.4:
            return "조건식 민감도를 낮추고 보조지표 필터를 강화하는 것을 검토하십시오."
        if avg_return < 0:
            return "손절 폭을 축소하거나 거래량 필터를 강화해 손실 거래를 줄이십시오."
        if win_rate > 0.6 and avg_return > 0.01:
            return "익절 폭을 소폭 확대하고 손절 폭을 완화해 추가 수익을 모색하십시오."
        return "현재 전략을 유지하면서 추가 표본을 확보해 통계적 유의성을 높이십시오."

    # ------------------------------------------------------------------ #
    # Public control
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        self.housekeeping_timer.stop()
        self.deactivate_condition()
        QApplication.instance().quit()


def load_config(path: str) -> OrderConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    try:
        config = OrderConfig(
            total_budget=int(raw.get("total_budget", 40_000_000)),
            max_positions=int(raw.get("max_positions", 4)),
            condition_name=str(raw["condition_name"]),
            condition_index=int(raw["condition_index"]),
            account_no=str(raw["account_no"]),
            screen_no=str(raw.get("screen_no", "2000")),
            price_unit=int(raw.get("price_unit", 1)),
            per_symbol_budget={str(k): int(v) for k, v in raw.get("per_symbol_budget", {}).items()},
            blocked_minutes=int(raw.get("blocked_minutes", 60)),
            trading_start=str(raw.get("trading_start", "09:20")),
            trading_end=str(raw.get("trading_end", "15:15")),
            profit_take_pct=float(raw.get("profit_take_pct", 0.01)),
            stop_loss_pct=float(raw.get("stop_loss_pct", 0.03)),
            trailing_drop_pct=float(raw.get("trailing_drop_pct", 0.01)),
            timeout_minutes=int(raw.get("timeout_minutes", 5)),
            banned_keywords=list(raw.get("banned_keywords", ["우", "ETF", "ETN", "스팩", "관리", "선물", "인버스"])),
            chart_minutes=int(raw.get("chart_minutes", 240)),
            volatility_lookback=int(raw.get("volatility_lookback", 20)),
            volatility_buffer=float(raw.get("volatility_buffer", 0.5)),
            adaptive_enabled=bool(raw.get("adaptive_enabled", True)),
            log_dir=str(raw.get("log_dir", "logs")),
            chart_dir=str(raw.get("chart_dir", "charts")),
        )
        if "banned_states" in raw:
            config.banned_states = set(map(str, raw["banned_states"]))
    except KeyError as exc:
        raise ValueError(f"Missing required config field: {exc}") from exc
    return config


def main() -> None:
    parser = ArgumentParser(description="Kiwoom condition auto trader (paper trading).")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    app = QApplication(sys.argv)
    trader = ConditionAutoTrader(config)
    trader.connect()

    try:
        available_conditions = trader.load_conditions()
        if config.condition_index not in available_conditions:
            raise ValueError(f"Condition index {config.condition_index} not found.")
        if available_conditions[config.condition_index] != config.condition_name:
            print(
                f"[WARN] Condition name mismatch. Using API value {available_conditions[config.condition_index]}",
                file=sys.stderr,
            )
            config.condition_name = available_conditions[config.condition_index]

        trader.activate_condition()
        print("[INFO] Condition subscription activated. Running event loop...")
        sys.exit(app.exec_())
    finally:
        trader.shutdown()


if __name__ == "__main__":
    main()

