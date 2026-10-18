# region imports
from AlgorithmImports import *
# endregion
"""Data-integrity check for the V2 workspace: minutes processed per UTC day (must be 1440, weekends
included) and the perpetual price LEAN sees at fixed UTC times, to compare with Deribit's own API."""
from deribit_common import DeribitVenue

CHECK_TIMES = ["2026-10-11 00:00", "2026-10-12 09:30", "2026-10-12 13:30", "2026-10-13 15:42",
               "2026-10-15 09:30", "2026-10-15 13:30", "2026-10-15 16:00"]


class DeribitDataCheck(QCAlgorithm):
    def Initialize(self):
        self.SetTimeZone(TimeZones.Utc)
        par = lambda k, d: self.GetParameter(k) or d
        self.coin = par("coin", "ETH")
        s, e = par("start", "2026-10-11"), par("end", "2026-10-15")
        self.SetStartDate(*map(int, s.split("-")))
        self.SetEndDate(*map(int, e.split("-")))
        self.SetCash(100000)
        self.venue = DeribitVenue(self, self.coin)
        self.u = self.venue.u
        self.minutes = {}
        self.marks = {datetime.strptime(t, "%Y-%m-%d %H:%M") for t in CHECK_TIMES}

    def OnData(self, data):
        if self.u not in data.Bars:
            return
        d = self.Time - timedelta(minutes=1)          # bar start time
        self.minutes[d.date()] = self.minutes.get(d.date(), 0) + 1
        if d in self.marks:
            self.Log(f"CHECK {d:%Y-%m-%d %a %H:%M} UTC bar open={data.Bars[self.u].Open} close={data.Bars[self.u].Close}")

    def OnEndOfAlgorithm(self):
        for day in sorted(self.minutes):
            self.Log(f"COVERAGE {day} {day:%a}: {self.minutes[day]} minute bars")
        self.Log(f"SUMMARY mode=DATA_CHECK coin={self.coin}")
