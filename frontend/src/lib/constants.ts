export const AIRLINE_NAMES: Record<string, string> = {
  "AA": "American Airlines",
  "DL": "Delta Air Lines",
  "UA": "United Airlines",
  "WN": "Southwest Airlines",
  "AS": "Alaska Airlines",
  "B6": "JetBlue Airways",
  "NK": "Spirit Airlines",
  "F9": "Frontier Airlines",
  "HA": "Hawaiian Airlines",
  "G4": "Allegiant Air",
  "YX": "Republic Airways",
  "OO": "SkyWest Airlines",
  "MQ": "Envoy Air",
  "9E": "Endeavor Air",
  "OH": "PSA Airlines",
  "YV": "Mesa Airlines",
  "EV": "ExpressJet Airlines",
  "VX": "Virgin America",
};

export const POPULAR_AIRPORTS: Array<{ code: string; name: string; city: string }> = [
  { code: "ATL", name: "Hartsfield-Jackson Atlanta", city: "Atlanta, GA" },
  { code: "DFW", name: "Dallas/Fort Worth International", city: "Dallas, TX" },
  { code: "DEN", name: "Denver International", city: "Denver, CO" },
  { code: "ORD", name: "O'Hare International", city: "Chicago, IL" },
  { code: "LAX", name: "Los Angeles International", city: "Los Angeles, CA" },
  { code: "CLT", name: "Charlotte Douglas International", city: "Charlotte, NC" },
  { code: "MCO", name: "Orlando International", city: "Orlando, FL" },
  { code: "LAS", name: "Harry Reid International", city: "Las Vegas, NV" },
  { code: "PHX", name: "Phoenix Sky Harbor", city: "Phoenix, AZ" },
  { code: "MIA", name: "Miami International", city: "Miami, FL" },
  { code: "SEA", name: "Seattle-Tacoma International", city: "Seattle, WA" },
  { code: "IAH", name: "George Bush Intercontinental", city: "Houston, TX" },
  { code: "JFK", name: "John F. Kennedy International", city: "New York, NY" },
  { code: "EWR", name: "Newark Liberty International", city: "Newark, NJ" },
  { code: "SFO", name: "San Francisco International", city: "San Francisco, CA" },
  { code: "MSP", name: "Minneapolis-St. Paul", city: "Minneapolis, MN" },
  { code: "DTW", name: "Detroit Metropolitan", city: "Detroit, MI" },
  { code: "BOS", name: "Logan International", city: "Boston, MA" },
  { code: "PHL", name: "Philadelphia International", city: "Philadelphia, PA" },
  { code: "LGA", name: "LaGuardia", city: "New York, NY" },
  { code: "BWI", name: "Baltimore/Washington", city: "Baltimore, MD" },
  { code: "SLC", name: "Salt Lake City International", city: "Salt Lake City, UT" },
  { code: "SAN", name: "San Diego International", city: "San Diego, CA" },
  { code: "IAD", name: "Washington Dulles", city: "Washington, DC" },
  { code: "DCA", name: "Ronald Reagan Washington", city: "Washington, DC" },
  { code: "TPA", name: "Tampa International", city: "Tampa, FL" },
  { code: "BNA", name: "Nashville International", city: "Nashville, TN" },
  { code: "AUS", name: "Austin-Bergstrom International", city: "Austin, TX" },
  { code: "MDW", name: "Chicago Midway", city: "Chicago, IL" },
  { code: "ANC", name: "Ted Stevens Anchorage", city: "Anchorage, AK" },
];

export const SAMPLE_ITINERARIES = [
  {
    label: "Morning Commuter (Low Risk)",
    airline_code: "DL",
    origin: "ATL",
    dest: "LGA",
    sched_dep_time: "07:30",
    sched_arr_time: "09:45",
  },
  {
    label: "Evening Hub Transit (High Risk)",
    airline_code: "AA",
    origin: "ORD",
    dest: "DFW",
    sched_dep_time: "18:45",
    sched_arr_time: "21:20",
  },
  {
    label: "Transcontinental (Average Risk)",
    airline_code: "UA",
    origin: "EWR",
    dest: "SFO",
    sched_dep_time: "11:15",
    sched_arr_time: "14:40",
  },
  {
    label: "Unseen Route (Geodesic Fallback)",
    airline_code: "DL",
    origin: "ANC",
    dest: "MIA",
    sched_dep_time: "08:00",
    sched_arr_time: "19:30",
  },
];

export const SAMPLE_CSV_CONTENT = `airline_code,origin,dest,flight_date,sched_dep_time,sched_arr_time
AA,DFW,ORD,2023-06-15,08:30,11:00
DL,ATL,LGA,2023-06-15,19:15,21:30
UA,ORD,SFO,2023-06-15,14:00,16:45
WN,DEN,PHX,2023-06-15,06:45,08:35
B6,BOS,MCO,2023-06-15,17:30,20:45
DL,ANC,MIA,2023-06-15,08:00,19:30
`;
