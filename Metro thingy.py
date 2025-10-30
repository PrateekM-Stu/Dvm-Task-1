import csv 
import os
import sys
import uuid
import json
from collections import deque, defaultdict
from datetime import datetime

"""Class Creation"""
class Station:
    """Represents a station."""
    def __init__(self, station_id: str, name: str):
        self.id = station_id
        self.name = name

    def to_csv_row(self):
        return [self.id, self.name]

    @staticmethod
    def from_csv_row(row):
        station_id = row[0]
        name = row[1] if len(row) > 1 else row[0]
        return Station(station_id, name)


class Line:
    """Represents a metro line (ordered station ids)."""
    def __init__(self, name: str, station_ids: list):
        self.name = name
        self.station_ids = list(station_ids)

    def to_csv_row(self):
        return [self.name, "|".join(self.station_ids)]

    @staticmethod
    def from_csv_row(row):
        name = row[0]
        station_str = row[1] if len(row) > 1 else ""
        station_ids = station_str.split("|") if station_str else []
        return Line(name, station_ids)


class Ticket:
    """Represents a purchased ticket."""
    def __init__(self, ticket_id: str, origin_id: str, destination_id: str, price: float, route_station_ids: list, instructions: list, created_at: str = None):
        self.ticket_id = ticket_id
        self.origin_id = origin_id
        self.destination_id = destination_id
        self.price = float(price)
        self.route_station_ids = list(route_station_ids)
        self.instructions = list(instructions)
        self.created_at = created_at or datetime.now().isoformat()

    def to_csv_row(self):
        return [
            self.ticket_id,
            self.origin_id,
            self.destination_id,
            f"{self.price:.2f}",
            "|".join(self.route_station_ids),
            json.dumps(self.instructions, ensure_ascii=False),
            self.created_at,
        ]

    @staticmethod
    def from_csv_row(row):
        ticket_id = row[0]
        origin_id = row[1] if len(row) > 1 else ""
        destination_id = row[2] if len(row) > 2 else ""
        price = float(row[3]) if len(row) > 3 and row[3] else 0.0
        route = row[4].split("|") if len(row) > 4 and row[4] else []
        instructions = json.loads(row[5]) if len(row) > 5 and row[5] else []
        created_at = row[6] if len(row) > 6 else None
        return Ticket(ticket_id, origin_id, destination_id, price, route, instructions, created_at)


# -------------------------------
# MetroSystem: holds stations, lines, tickets, and graph logic
# -------------------------------
class MetroSystem:
    def __init__(self, stations_file='stations.csv', lines_file='lines.csv', tickets_file='tickets.csv'):
        self.stations_file = stations_file
        self.lines_file = lines_file
        self.tickets_file = tickets_file

        self.stations = {}  # id -> Station
        self.lines = {}     # name -> Line
        self.tickets = {}   # ticket_id -> Ticket
        self.graph = defaultdict(set)  # adjacency list: station_id -> set(station_id)

        self._ensure_csvs_exist()
        self._load_stations()
        self._load_lines()
        self._build_graph()
        self._load_tickets()

    # -------------------------------
    # CSV bootstrapping and I/O
    # -------------------------------
    def _ensure_csvs_exist(self):
        # If either stations or lines file is missing, create example webbed network
        if not os.path.exists(self.stations_file) or not os.path.exists(self.lines_file):
            print('Station/line CSV missing - creating example webbed network CSVs...')
            self._create_example_network()

        # Ensure tickets CSV exists with header
        if not os.path.exists(self.tickets_file):
            with open(self.tickets_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ticket_id', 'origin_id', 'destination_id', 'price', 'route_station_ids', 'instructions_json', 'created_at'])

    def _create_example_network(self):
        example_lines = {
            'Red': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
            'Blue': ['B1', 'B2', 'R3', 'B4', 'B5', 'B6'],
            'Green': ['G1', 'G2', 'R5', 'G4', 'G5', 'G6'],
            'Yellow': ['Y1', 'B5', 'R5', 'G4', 'Y4']
        }

        # build a station list with human-friendly names
        stations = {}
        for line_name, sids in example_lines.items():
            for sid in sids:
                if sid not in stations:
                    stations[sid] = f"{sid} Station"

        # write stations.csv
        with open(self.stations_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['station_id', 'station_name'])
            for sid, name in sorted(stations.items()):
                writer.writerow([sid, name])

        # write lines.csv
        with open(self.lines_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['line_name', 'station_id_sequence'])
            for line_name, sids in example_lines.items():
                writer.writerow([line_name, '|'.join(sids)])

        print('Example stations.csv and lines.csv created.')

    def _load_stations(self):
        if not os.path.exists(self.stations_file):
            return
        with open(self.stations_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                try:
                    st = Station.from_csv_row(row)
                    self.stations[st.id] = st
                except Exception:
                    # skip malformed rows
                    continue

    def _load_lines(self):
        if not os.path.exists(self.lines_file):
            return
        with open(self.lines_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row or len(row) < 1:
                    continue
                try:
                    line = Line.from_csv_row(row)
                    self.lines[line.name] = line
                except Exception:
                    continue

    def _load_tickets(self):
        if not os.path.exists(self.tickets_file):
            return
        with open(self.tickets_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                try:
                    t = Ticket.from_csv_row(row)
                    self.tickets[t.ticket_id] = t
                except Exception:
                    # skip malformed ticket rows
                    continue

    def _build_graph(self):
        self.graph.clear()
        for line in self.lines.values():
            s = line.station_ids
            for i in range(len(s)-1):
                a, b = s[i], s[i+1]
                self.graph[a].add(b)
                self.graph[b].add(a)

    # -------------------------------
    # Utility & lookup
    # -------------------------------
    def list_stations(self):
        return sorted([(sid, st.name) for sid, st in self.stations.items()], key=lambda x: x[0])

    def station_exists(self, station_id):
        return station_id in self.stations

    def station_name(self, station_id):
        return self.stations[station_id].name if station_id in self.stations else station_id

    def find_station_by_name(self, partial_name):
        q = partial_name.lower()
        return [sid for sid, st in self.stations.items() if q in st.name.lower()]

    # -------------------------------
    # Shortest path (BFS). Returns list of station IDs or None
    # -------------------------------
    def shortest_path(self, origin_id, destination_id):
        if origin_id == destination_id:
            return [origin_id]
        if origin_id not in self.graph or destination_id not in self.graph:
            return None

        visited = set()
        parent = {}
        q = deque([origin_id])
        visited.add(origin_id)

        found = False
        while q:
            cur = q.popleft()
            if cur == destination_id:
                found = True
                break
            for nb in self.graph[cur]:
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    q.append(nb)
        if not found:
            return None
        path = [destination_id]
        while path[-1] != origin_id:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    # -------------------------------
    # Path instructions: identify line segments and transfers
    # -------------------------------
    def path_instructions(self, path_station_ids):
        if not path_station_ids or len(path_station_ids) == 1:
            return ["You are already at your destination."]

        # Precompute adjacency -> lines mapping
        pair_lines = defaultdict(list)  # (a,b) -> [line names]
        for line_name, line in self.lines.items():
            s = line.station_ids
            for i in range(len(s)-1):
                a, b = s[i], s[i+1]
                pair_lines[(a,b)].append(line_name)
                pair_lines[(b,a)].append(line_name)

        instructions = []
        # determine best line for each edge preferring continuity
        current_line = None
        seg_start = path_station_ids[0]

        for i in range(len(path_station_ids)-1):
            a = path_station_ids[i]
            b = path_station_ids[i+1]
            lines_here = pair_lines.get((a,b), [])
            chosen = None
            if current_line and current_line in lines_here:
                chosen = current_line
            elif lines_here:
                # pick the line that will maximize continuity ahead if possible
                chosen = lines_here[0]
                # try to pick a line that also covers next hop (lookahead)
                if i+1 < len(path_station_ids)-1:
                    c = path_station_ids[i+2]
                    for ln in lines_here:
                        if ln in pair_lines.get((b,c), []):
                            chosen = ln
                            break
            # if current_line is None, start a new segment
            if current_line is None:
                current_line = chosen
                seg_start = a
            # if chosen differs from current_line, we must end previous segment and record transfer
            if chosen != current_line:
                seg_end = a
                instructions.append(f"Take {current_line or 'a line'} from {self.station_name(seg_start)} (ID:{seg_start}) to {self.station_name(seg_end)} (ID:{seg_end}).")
                instructions.append(f"Change lines at {self.station_name(a)} (ID:{a}).")
                current_line = chosen
                seg_start = a
        # finish final segment
        seg_end = path_station_ids[-1]
        instructions.append(f"Take {current_line or 'a line'} from {self.station_name(seg_start)} (ID:{seg_start}) to {self.station_name(seg_end)} (ID:{seg_end}).")

        # compress repeated change lines
        cleaned = []
        for instr in instructions:
            if cleaned and cleaned[-1] == instr and instr.startswith('Change'):
                continue
            cleaned.append(instr)
        return cleaned

    # -------------------------------
    # Pricing: base fare per hop (station-to-station)
    # -------------------------------
    def price_for_path(self, path_station_ids):
        BASE_FARE_PER_STATION = 5.0
        edges = max(0, len(path_station_ids)-1)
        return round(edges * BASE_FARE_PER_STATION, 2)

    # -------------------------------
    # Ticket creation and persistence
    # -------------------------------
    def create_ticket(self, origin_id, destination_id):
        path = self.shortest_path(origin_id, destination_id)
        if path is None:
            raise ValueError('No path between these stations')
        price = self.price_for_path(path)
        instructions = self.path_instructions(path)
        ticket_id = str(uuid.uuid4())
        ticket = Ticket(ticket_id, origin_id, destination_id, price, path, instructions)
        self.tickets[ticket_id] = ticket
        # append to CSV
        with open(self.tickets_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(ticket.to_csv_row())
        return ticket

    def list_tickets(self):
        return sorted(self.tickets.values(), key=lambda t: t.created_at, reverse=True)

    # -------------------------------
    # Visualization (lazy imports)
    # -------------------------------
    def visualize(self, highlight_path=None):
        # Try to import visualization libraries only when asked to visualize
        try:
            import matplotlib.pyplot as plt
            HAS_MATPLOTLIB = True
        except Exception:
            HAS_MATPLOTLIB = False

        try:
            import networkx as nx
            HAS_NETWORKX = True
        except Exception:
            HAS_NETWORKX = False

        if not HAS_MATPLOTLIB:
            print('Visualization unavailable: install matplotlib (and networkx for enhanced graphs) to enable this feature.')
            print('Install with: pip install matplotlib networkx')
            return

        # Build graph for plotting
        if HAS_NETWORKX:
            G = nx.Graph()
            for sid in self.stations:
                G.add_node(sid)
            for a, neighbors in self.graph.items():
                for b in neighbors:
                    if a < b:
                        G.add_edge(a, b)

            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)

            # base nodes/edges
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightgray')
            nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray')
            nx.draw_networkx_labels(G, pos, {sid: self.stations[sid].name for sid in G.nodes()}, font_size=9)

            if highlight_path:
                edges = list(zip(highlight_path, highlight_path[1:]))
                nx.draw_networkx_nodes(G, pos, nodelist=highlight_path, node_color='orange', node_size=800)
                nx.draw_networkx_edges(G, pos, edgelist=edges, width=3, edge_color='red')

            plt.title('Metro Map')
            plt.axis('off')
            plt.show()
            return

        # If networkx isn't available, use crude matplotlib plotting (circular layout)
        n = len(self.stations)
        ids = list(self.stations.keys())
        import math
        angles = [2 * math.pi * i / max(1, n) for i in range(n)]
        coords = {ids[i]: (math.cos(angles[i]), math.sin(angles[i])) for i in range(n)}

        plt.figure(figsize=(8, 8))
        for a, neighbors in self.graph.items():
            for b in neighbors:
                if a < b:
                    ax = [coords[a][0], coords[b][0]]
                    ay = [coords[a][1], coords[b][1]]
                    plt.plot(ax, ay, color='gray')
        xs = [coords[i][0] for i in ids]
        ys = [coords[i][1] for i in ids]
        plt.scatter(xs, ys)
        for i, sid in enumerate(ids):
            plt.text(xs[i], ys[i], f"{sid} {self.stations[sid].name}")
        plt.title('Metro Map (approx)')
        plt.axis('off')
        plt.show()


# -------------------------------
# Command-line interface
# -------------------------------
class CLI:
    def __init__(self, metro_system: MetroSystem):
        self.metro = metro_system

    def run(self):
        print('Welcome to the Metro Ticket Purchasing System')
        while True:
            print('Options:')
            print('1) List stations')
            print('2) Purchase ticket')
            print('3) View purchased tickets')
            print('4) Visualize map')
            print('5) Find shortest route (no purchase)')
            print('6) Exit')
            try:
                ch = input('> ').strip()
            except (KeyboardInterrupt, EOFError):
                print('Interrupted. Exiting...')
                return
            if ch == '1':
                self.cmd_list_stations()
            elif ch == '2':
                self.cmd_purchase()
            elif ch == '3':
                self.cmd_view_tickets()
            elif ch == '4':
                self.cmd_visualize()
            elif ch == '5':
                self.cmd_find_route()
            elif ch == '6':
                print('Goodbye!')
                break
            else:
                print('Invalid choice.')

    def cmd_list_stations(self):
        rows = self.metro.list_stations()
        print('Stations (ID : Name):')
        for sid, name in rows:
            print(f"{sid} : {name}")

    def _ask_station(self, prompt):
        while True:
            try:
                q = input(prompt + ' (enter station id or substring of name): ').strip()
            except (KeyboardInterrupt, EOFError):
                print('Interrupted. Exiting...')
                raise
            if not q:
                print('Please enter something.')
                continue
            if self.metro.station_exists(q):
                return q
            candidates = self.metro.find_station_by_name(q)
            if len(candidates) == 1:
                sid = candidates[0]
                print(f"Interpreting as {sid} : {self.metro.station_name(sid)}")
                return sid
            elif len(candidates) > 1:
                print('Multiple matches:')
                for c in candidates:
                    print(f"{c} : {self.metro.station_name(c)}")
                print('Please type the station ID you want.')
                continue
            else:
                print('No matches found. Try again.')

    def cmd_purchase(self):
        print('Purchase ticket')
        try:
            origin = self._ask_station('Origin')
            destination = self._ask_station('Destination')
        except Exception:
            return
        if origin == destination:
            print('Origin and destination are the same — no ticket needed.')
            return
        path = self.metro.shortest_path(origin, destination)
        if path is None:
            print('No route found between these stations.')
            return
        price = self.metro.price_for_path(path)
        instructions = self.metro.path_instructions(path)

        print('Route:')
        print(' -> '.join([f"{sid}({self.metro.station_name(sid)})" for sid in path]))
        print(f'Price: {price:.2f}')
        print('Instructions:')
        for line in instructions:
            print('-', line)

        try:
            ok = input('Confirm purchase? (y/n): ').strip().lower()
        except (KeyboardInterrupt, EOFError):
            print('Interrupted. Purchase cancelled.')
            return
        if ok == 'y':
            try:
                ticket = self.metro.create_ticket(origin, destination)
                print('Ticket purchased:')
                self._print_ticket(ticket)
            except Exception as e:
                print('Failed to create ticket:', e)
        else:
            print('Purchase cancelled.')

    def cmd_view_tickets(self):
        ts = self.metro.list_tickets()
        if not ts:
            print('No tickets purchased yet.')
            return
        for t in ts:
            self._print_ticket(t)
            print('---')

    def cmd_visualize(self):
        try:
            choice = input('Do you want to highlight a route? (y/n): ').strip().lower()
        except (KeyboardInterrupt, EOFError):
            print('4' \
            'Interrupted.')
            return
        if choice == 'y':
            try:
                origin = self._ask_station('Origin')
                dest = self._ask_station('Destination')
            except Exception:
                return
            path = self.metro.shortest_path(origin, dest)
            if not path:
                print('No route found.')
                return
            self.metro.visualize(highlight_path=path)
        else:
            self.metro.visualize()

    def cmd_find_route(self):
        try:
            origin = self._ask_station('Origin')
            dest = self._ask_station('Destination')
        except Exception:
            return
        path = self.metro.shortest_path(origin, dest)
        if not path:
            print('No route found.')
            return
        print('Shortest route:')
        print(' -> '.join([f"{sid}({self.metro.station_name(sid)})" for sid in path]))
        print('Instructions:')
        for i in self.metro.path_instructions(path):
            print('-', i)

    def _print_ticket(self, ticket: Ticket):
        print(f"Ticket ID: {ticket.ticket_id}")
        print(f"From: {ticket.origin_id} : {self.metro.station_name(ticket.origin_id)}")
        print(f"To:   {ticket.destination_id} : {self.metro.station_name(ticket.destination_id)}")
        print(f"Price: {ticket.price:.2f}")
        print(f"Route: {' -> '.join([f'{sid}({self.metro.station_name(sid)})' for sid in ticket.route_station_ids])}")
        print('Instructions:')
        for instr in ticket.instructions:
            print('-', instr)
        print(f"Purchased at: {ticket.created_at}")


# -------------------------------
# Entry point
# -------------------------------
if __name__ == '__main__':
    
    metro = MetroSystem()
    cli = CLI(metro)
    try:
        cli.run()
    except KeyboardInterrupt:
        print('Interrupted. Exiting...')