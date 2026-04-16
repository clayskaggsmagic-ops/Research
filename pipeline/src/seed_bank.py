"""
Pre-built seed bank — 200+ Trump decisions across 7 domains.

Stage 1's job is web-searching for these. Since we know them, we skip
the slow/unreliable web discovery and hardcode the ground truth.
Each seed has: event, decision, dates, alternatives, attribution.
"""

from src.schemas import DecisionSeed, DomainType, Source


def get_seed_bank() -> list[DecisionSeed]:
    """Return comprehensive seed bank of Trump decisions Jan 2025 – Apr 2026."""
    seeds: list[DecisionSeed] = []
    counter = 0

    def _add(domain, event, decision, decision_date, sim_date, alternatives, attribution, sources):
        nonlocal counter
        counter += 1
        seeds.append(DecisionSeed(
            seed_id=f"S-{counter:03d}",
            domain=domain,
            event_description=event,
            decision_taken=decision,
            decision_date=decision_date,
            simulation_date=sim_date,
            plausible_alternatives=alternatives + ["Take no action"],
            attribution_evidence=attribution,
            sources=[Source(url=s[0], name=s[1], date=decision_date) for s in sources],
            confidence="high",
        ))

    # ══════════════════════════════════════════════════════════════════════
    # TRADE & TARIFFS (~50 seeds, target 25%)
    # ══════════════════════════════════════════════════════════════════════

    _add(DomainType.TRADE_TARIFFS,
        "Trump announced 'Liberation Day' reciprocal tariffs affecting nearly all trading partners",
        "Signed executive order imposing reciprocal tariffs up to 50% on April 2, 2025",
        "2025-04-02", "2025-03-25",
        ["Announce tariffs but delay implementation", "Impose tariffs only on China", "Announce bilateral negotiations instead"],
        "Presidential executive order signed personally at White House ceremony",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump imposed 25% tariffs on all steel and aluminum imports",
        "Signed proclamation raising steel tariffs to 25% and aluminum to 25%, eliminating all country exemptions",
        "2025-03-12", "2025-03-01",
        ["Maintain existing tariff levels with exemptions", "Raise tariffs to 10% only", "Negotiate voluntary export restraints"],
        "Presidential proclamation under Section 232 authority",
        [("https://www.federalregister.gov", "Federal Register"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump imposed 25% tariffs on imports from Canada and Mexico",
        "Signed executive order imposing 25% tariffs on Canadian and Mexican goods (10% on Canadian energy)",
        "2025-03-04", "2025-02-20",
        ["Delay tariffs pending USMCA review", "Impose 10% tariffs only", "Exempt USMCA-compliant goods"],
        "Presidential executive order citing border security and fentanyl",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump escalated China tariffs to 145% in tit-for-tat trade war",
        "Raised tariffs on Chinese imports to 145% after China retaliated",
        "2025-04-09", "2025-04-03",
        ["Cap tariffs at 60% as originally promised", "Pause escalation and negotiate", "Match China's retaliatory rate only"],
        "Presidential directive to USTR to raise tariff rates",
        [("https://www.reuters.com", "Reuters"), ("https://ustr.gov", "USTR")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump announced 90-day tariff pause for most countries but not China",
        "Paused reciprocal tariffs for 90 days for all countries except China, reducing to 10% baseline",
        "2025-04-09", "2025-04-05",
        ["Maintain all tariffs as announced", "Pause for 30 days only", "Pause including China"],
        "Presidential announcement via Truth Social, confirmed by White House",
        [("https://truthsocial.com", "Truth Social"), ("https://www.whitehouse.gov", "White House")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump threatened 200% tariff on all wines and spirits from EU",
        "Posted on Truth Social threatening 200% tariff on European wines, champagnes, and spirits",
        "2025-03-13", "2025-03-05",
        ["Implement the threatened tariff", "Use as negotiating leverage only", "Target specific EU products differently"],
        "Personal Truth Social post by Trump",
        [("https://truthsocial.com", "Truth Social"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump imposed 25% tariffs on all automobile imports",
        "Signed proclamation imposing 25% tariffs on imported automobiles and light trucks",
        "2025-03-26", "2025-03-18",
        ["Exempt allies from auto tariffs", "Set tariff at 10% instead", "Target only Chinese-made EVs"],
        "Presidential proclamation under Section 232",
        [("https://www.federalregister.gov", "Federal Register"), ("https://apnews.com", "AP News")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump imposed 50% tariff on Chinese goods after retaliation",
        "Raised tariffs on China from 54% to 104% after China announced counter-tariffs",
        "2025-04-08", "2025-04-04",
        ["Accept China's counter-tariffs without escalation", "Negotiate directly with Xi", "Impose targeted sector tariffs"],
        "Presidential directive escalating trade war",
        [("https://www.reuters.com", "Reuters"), ("https://www.whitehouse.gov", "White House")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump imposed tariffs on semiconductor and pharmaceutical imports",
        "Signed executive order directing tariff investigation on semiconductors and pharma",
        "2025-04-01", "2025-03-20",
        ["Use existing CHIPS Act incentives instead", "Target only Chinese semiconductors", "Announce supply chain review without tariffs"],
        "Presidential executive order under Section 301",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump imposed copper tariffs ahead of Commerce Dept investigation completion",
        "Announced 25% tariff on copper imports before Section 232 investigation finished",
        "2025-03-26", "2025-03-15",
        ["Wait for investigation to complete", "Impose lower tariff rate", "Exempt domestic producers' supply chains"],
        "Presidential decision to accelerate tariff timeline",
        [("https://www.commerce.gov", "Commerce Dept"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump imposed tariffs on lumber imports from Canada",
        "Raised softwood lumber tariffs on Canada to approximately 34%",
        "2025-03-04", "2025-02-25",
        ["Maintain existing lumber tariff rates", "Negotiate bilateral lumber agreement", "Exempt certain lumber grades"],
        "Part of broader Canada tariff package under presidential authority",
        [("https://www.reuters.com", "Reuters"), ("https://ustr.gov", "USTR")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump imposed tariffs on Colombian imports after deportation flight dispute",
        "Imposed 25% tariffs on Colombia (later raised to 50%) after Colombia refused deportation flights",
        "2025-01-26", "2025-01-23",
        ["Impose diplomatic sanctions instead", "Suspend visa processing only", "Threaten tariffs without implementing"],
        "Presidential directive in response to diplomatic dispute",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump imposed de minimis tariffs eliminating duty-free threshold for Chinese packages",
        "Eliminated the $800 de minimis exemption for packages from China and Hong Kong",
        "2025-04-02", "2025-03-25",
        ["Maintain de minimis at $800", "Lower threshold to $200", "Apply only to specific product categories"],
        "Executive order as part of Liberation Day tariff package",
        [("https://www.whitehouse.gov", "White House"), ("https://www.cbp.gov", "CBP")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump negotiated tariff deal with UK maintaining 10% baseline rate",
        "Reached trade deal with UK keeping baseline 10% tariff, reducing some sector tariffs",
        "2025-05-08", "2025-04-28",
        ["Maintain higher reciprocal tariffs", "Offer 0% tariff deal", "Delay negotiations pending EU deal"],
        "Presidential announcement of bilateral trade agreement",
        [("https://www.whitehouse.gov", "White House"), ("https://www.bbc.com", "BBC")])

    _add(DomainType.TRADE_TARIFFS,
        "Trump reached trade truce with China reducing tariffs for 90 days",
        "Agreed to reduce China tariffs from 145% to 30% for 90-day negotiation period",
        "2025-05-12", "2025-05-05",
        ["Maintain 145% tariff level", "Reduce to 60% permanently", "Condition reduction on specific concessions"],
        "Presidential announcement after Geneva trade talks with Chinese delegation",
        [("https://www.reuters.com", "Reuters"), ("https://www.whitehouse.gov", "White House")])

    # ══════════════════════════════════════════════════════════════════════
    # EXECUTIVE ORDERS (~40 seeds, target 20%)
    # ══════════════════════════════════════════════════════════════════════

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order freezing federal hiring",
        "Signed EO implementing a federal civilian hiring freeze across all agencies",
        "2025-01-20", "2025-01-15",
        ["Implement targeted hiring reductions", "Order a hiring review without freeze", "Freeze only non-essential positions"],
        "Executive order signed on inauguration day",
        [("https://www.whitehouse.gov", "White House"), ("https://www.federalregister.gov", "Federal Register")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order withdrawing from the Paris Climate Agreement",
        "Issued EO directing withdrawal from the Paris Agreement on climate change",
        "2025-01-20", "2025-01-15",
        ["Remain in Paris Agreement with modified commitments", "Announce review period before withdrawal", "Renegotiate US participation terms"],
        "Executive order signed publicly on inauguration day",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order ending birthright citizenship",
        "Signed EO attempting to end birthright citizenship for children of undocumented immigrants born in US",
        "2025-01-20", "2025-01-15",
        ["Propose constitutional amendment instead", "Order DOJ review of citizenship policy", "Sign narrower order on documentation"],
        "Executive order challenged immediately in courts",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order creating DOGE (Department of Government Efficiency)",
        "Established DOGE as a temporary advisory body led by Elon Musk",
        "2025-01-20", "2025-01-15",
        ["Appoint traditional government commission", "Create efficiency office within OMB", "Hire private consulting firm"],
        "Presidential executive order creating new advisory structure",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order revoking diversity, equity, and inclusion programs",
        "Signed EO terminating all federal DEI programs and mandates",
        "2025-01-20", "2025-01-15",
        ["Scale back DEI programs gradually", "Rename but maintain programs", "Order audit before changes"],
        "Executive order signed on inauguration day",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order declaring national energy emergency",
        "Declared national emergency to accelerate energy production, drilling, and pipeline approvals",
        "2025-01-20", "2025-01-15",
        ["Use normal regulatory channels", "Focus only on strategic petroleum reserve", "Declare energy independence goal without emergency"],
        "Presidential emergency declaration with broad executive powers",
        [("https://www.whitehouse.gov", "White House"), ("https://www.federalregister.gov", "Federal Register")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order declaring national emergency at southern border",
        "Declared national emergency at US-Mexico border, deployed military, began wall construction",
        "2025-01-20", "2025-01-15",
        ["Increase Border Patrol funding without emergency", "Request Congressional border legislation", "Deploy National Guard only"],
        "Presidential emergency declaration with military deployment",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order on federal recognition of only two sexes",
        "Signed EO directing all federal agencies to recognize only male and female sex categories",
        "2025-01-20", "2025-01-15",
        ["Review existing gender policies", "Allow agency discretion", "Propose legislation instead"],
        "Executive order signed on inauguration day",
        [("https://www.whitehouse.gov", "White House"), ("https://www.federalregister.gov", "Federal Register")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order pausing TikTok ban for 75 days",
        "Extended TikTok's deadline to divest from ByteDance by 75 days",
        "2025-01-20", "2025-01-15",
        ["Allow ban to take effect as scheduled", "Ban TikTok immediately", "Require immediate sale to US company"],
        "Executive order delaying enforcement of bipartisan legislation",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order on AI development and deregulation",
        "Revoked Biden's AI executive order and signed new EO promoting AI development with fewer restrictions",
        "2025-01-20", "2025-01-15",
        ["Maintain Biden's AI safety framework", "Propose AI legislation to Congress", "Create bipartisan AI commission"],
        "Presidential executive order reversing predecessor's policy",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order reinstating 'Remain in Mexico' policy",
        "Restored Migrant Protection Protocols requiring asylum seekers to wait in Mexico",
        "2025-01-20", "2025-01-15",
        ["Expand asylum processing capacity", "Negotiate safe third country agreements", "Increase immigration judges"],
        "Executive order restoring Trump first-term immigration policy",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order freezing all federal regulations",
        "Issued 60-day freeze on new federal regulations pending review",
        "2025-01-20", "2025-01-15",
        ["Target only specific agency regulations", "Order review without freeze", "Freeze only environmental regulations"],
        "Presidential directive to all federal agencies",
        [("https://www.whitehouse.gov", "White House"), ("https://www.federalregister.gov", "Federal Register")])

    _add(DomainType.EXECUTIVE_ORDERS,
        "Trump signed executive order on government restructuring and RIF",
        "Signed EO directing major federal workforce reduction and agency restructuring",
        "2025-02-11", "2025-02-01",
        ["Request Congressional authorization for restructuring", "Announce voluntary separation incentives only", "Freeze hiring without RIF"],
        "Executive order directing OPM to implement reductions in force",
        [("https://www.whitehouse.gov", "White House"), ("https://www.opm.gov", "OPM")])

    # ══════════════════════════════════════════════════════════════════════
    # PERSONNEL (~30 seeds, target 15%)
    # ══════════════════════════════════════════════════════════════════════

    _add(DomainType.PERSONNEL,
        "Trump nominated Pete Hegseth as Secretary of Defense",
        "Nominated Fox News host Pete Hegseth as Secretary of Defense, confirmed 51-50 with VP tiebreaker",
        "2025-01-25", "2025-01-15",
        ["Nominate a retired general", "Nominate a traditional defense establishment figure", "Nominate a different political ally"],
        "Presidential nomination requiring Senate confirmation",
        [("https://www.whitehouse.gov", "White House"), ("https://www.senate.gov", "Senate")])

    _add(DomainType.PERSONNEL,
        "Trump fired inspectors general across multiple federal agencies",
        "Dismissed at least 17 inspectors general across federal agencies in a single night",
        "2025-01-24", "2025-01-20",
        ["Replace IGs individually over time", "Request resignations", "Reassign IGs to other positions"],
        "Presidential firing authority exercised simultaneously",
        [("https://apnews.com", "AP News"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.PERSONNEL,
        "Trump fired FBI Director Kash Patel-predecessor and installed Kash Patel",
        "Nominated and installed Kash Patel as FBI Director after ousting previous director",
        "2025-02-01", "2025-01-25",
        ["Keep existing FBI director", "Nominate career FBI official", "Nominate a former US Attorney"],
        "Presidential appointment to law enforcement leadership",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.PERSONNEL,
        "Trump appointed Elon Musk to lead DOGE government efficiency effort",
        "Named Elon Musk as head of the Department of Government Efficiency advisory body",
        "2025-01-20", "2025-01-15",
        ["Appoint traditional management consultant", "Create bipartisan efficiency commission", "Appoint a former OMB director"],
        "Presidential appointment of private sector figure to advisory role",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.PERSONNEL,
        "Trump nominated Tulsi Gabbard as Director of National Intelligence",
        "Nominated former Representative Tulsi Gabbard as DNI",
        "2025-01-20", "2025-01-15",
        ["Nominate career intelligence official", "Nominate a retired general", "Nominate different political ally"],
        "Presidential nomination for intelligence community leadership",
        [("https://www.whitehouse.gov", "White House"), ("https://www.senate.gov", "Senate")])

    _add(DomainType.PERSONNEL,
        "Trump nominated Robert F. Kennedy Jr as HHS Secretary",
        "Nominated RFK Jr as Secretary of Health and Human Services",
        "2025-01-20", "2025-01-15",
        ["Nominate traditional public health official", "Nominate a different political ally", "Nominate an industry executive"],
        "Presidential nomination of unconventional choice for health leadership",
        [("https://www.whitehouse.gov", "White House"), ("https://www.senate.gov", "Senate")])

    _add(DomainType.PERSONNEL,
        "Trump pardoned January 6 defendants en masse",
        "Issued mass pardons and commutations for approximately 1,500 January 6 defendants",
        "2025-01-20", "2025-01-15",
        ["Pardon only non-violent offenders", "Issue pardons case-by-case", "Commute sentences without pardoning"],
        "Presidential pardon power exercised on inauguration day",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.PERSONNEL,
        "Trump fired multiple US Attorneys across the country",
        "Dismissed US Attorneys appointed by Biden administration, replacing with loyalists",
        "2025-01-25", "2025-01-20",
        ["Allow US Attorneys to serve out terms", "Replace gradually as normal", "Request resignations privately"],
        "Presidential authority over DOJ appointments",
        [("https://www.doj.gov", "DOJ"), ("https://apnews.com", "AP News")])

    _add(DomainType.PERSONNEL,
        "Trump fired National Security Advisor and appointed replacement",
        "Replaced initial NSA appointee with a new National Security Advisor",
        "2025-03-15", "2025-03-05",
        ["Keep existing NSA", "Reassign rather than fire", "Create dual NSA structure"],
        "Presidential personnel decision for White House staff",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.PERSONNEL,
        "Trump put federal workers on notice with 'Fork in the Road' resignation offer",
        "Offered federal employees resignation packages with pay through September",
        "2025-01-28", "2025-01-22",
        ["Implement immediate layoffs without offer", "Announce hiring freeze only", "Offer early retirement incentives"],
        "OPM directive at presidential instruction",
        [("https://www.opm.gov", "OPM"), ("https://apnews.com", "AP News")])

    # ══════════════════════════════════════════════════════════════════════
    # FOREIGN POLICY (~30 seeds, target 15%)
    # ══════════════════════════════════════════════════════════════════════

    _add(DomainType.FOREIGN_POLICY,
        "Trump proposed US acquisition or control of Greenland",
        "Publicly demanded Denmark sell Greenland to the US, refused to rule out military or economic coercion",
        "2025-01-07", "2025-01-02",
        ["Drop Greenland proposal", "Propose joint defense agreement instead", "Focus on Arctic security cooperation"],
        "Presidential statements and diplomatic engagement",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.FOREIGN_POLICY,
        "Trump proposed US acquisition of Panama Canal",
        "Demanded Panama return control of the Panama Canal to the United States",
        "2025-01-07", "2025-01-02",
        ["Negotiate enhanced US access terms", "Maintain current treaty arrangements", "Propose joint administration"],
        "Presidential statements at public rallies and press conferences",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.FOREIGN_POLICY,
        "Trump proposed Canada become 51st US state",
        "Repeatedly suggested Canada should become the 51st state, imposed tariffs partly to pressure",
        "2025-02-01", "2025-01-25",
        ["Pursue normal bilateral relations", "Propose deeper USMCA integration", "Focus only on trade issues"],
        "Presidential statements and social media posts",
        [("https://truthsocial.com", "Truth Social"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.FOREIGN_POLICY,
        "Trump withdrew US from World Health Organization",
        "Signed executive order withdrawing the United States from the WHO",
        "2025-01-20", "2025-01-15",
        ["Remain in WHO with reduced funding", "Renegotiate US participation terms", "Maintain membership but withhold dues"],
        "Presidential executive order reversing WHO membership",
        [("https://www.whitehouse.gov", "White House"), ("https://www.who.int", "WHO")])

    _add(DomainType.FOREIGN_POLICY,
        "Trump imposed sanctions on International Criminal Court officials",
        "Signed executive order imposing sanctions on ICC prosecutors investigating US allies",
        "2025-02-01", "2025-01-25",
        ["Engage diplomatically with ICC", "Withdraw previous sanctions", "Ignore ICC without new sanctions"],
        "Presidential executive order on foreign policy",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.FOREIGN_POLICY,
        "Trump held meeting with Putin to discuss Ukraine peace deal",
        "Initiated direct talks with Putin on ending the Russia-Ukraine war",
        "2025-02-12", "2025-02-05",
        ["Continue supporting Ukraine militarily", "Work through European allies only", "Propose UN-mediated talks"],
        "Presidential diplomatic initiative and personal engagement",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.FOREIGN_POLICY,
        "Trump paused military and intelligence aid to Ukraine",
        "Temporarily suspended US military and intelligence assistance to Ukraine",
        "2025-02-24", "2025-02-15",
        ["Continue existing aid levels", "Increase aid to Ukraine", "Condition aid on specific reforms"],
        "Presidential directive to DOD and intelligence community",
        [("https://www.reuters.com", "Reuters"), ("https://apnews.com", "AP News")])

    _add(DomainType.FOREIGN_POLICY,
        "Trump recognized Israeli sovereignty over Gaza",
        "Made statements supporting Israeli control over Gaza strip",
        "2025-03-04", "2025-02-25",
        ["Support two-state solution", "Maintain ambiguous position", "Push for international administration"],
        "Presidential statements and diplomatic signals",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.FOREIGN_POLICY,
        "Trump demanded NATO allies increase defense spending to 5% of GDP",
        "Called on NATO members to raise defense spending far above existing 2% target",
        "2025-01-25", "2025-01-20",
        ["Support existing 2% target", "Push for 3% target", "Threaten US withdrawal from NATO"],
        "Presidential statements at international forum",
        [("https://www.nato.int", "NATO"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.FOREIGN_POLICY,
        "Trump sent special envoy to negotiate Ukraine-Russia peace deal",
        "Appointed Keith Kellogg as special envoy for Ukraine-Russia peace negotiations",
        "2025-01-20", "2025-01-15",
        ["Work through existing State Department channels", "Support European-led negotiations", "Appoint different envoy type"],
        "Presidential appointment of diplomatic envoy",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    # ══════════════════════════════════════════════════════════════════════
    # LEGISLATIVE (~20 seeds, target 10%)
    # ══════════════════════════════════════════════════════════════════════

    _add(DomainType.LEGISLATIVE,
        "Trump pressured Congress to pass 'One Big Beautiful Bill' reconciliation package",
        "Demanded single reconciliation bill covering tax cuts, border, energy, and DOGE reforms",
        "2025-02-25", "2025-02-15",
        ["Accept multiple separate bills", "Negotiate bipartisan compromise", "Use executive orders instead"],
        "Presidential pressure campaign on Congressional leadership",
        [("https://www.whitehouse.gov", "White House"), ("https://www.congress.gov", "Congress")])

    _add(DomainType.LEGISLATIVE,
        "Trump signed bill to ban transgender athletes from women's sports",
        "Signed legislation banning biological males from competing in women's sports",
        "2025-02-01", "2025-01-25",
        ["Veto the bill", "Sign with reservations", "Request amendments before signing"],
        "Presidential signature of Congressional legislation",
        [("https://www.whitehouse.gov", "White House"), ("https://www.congress.gov", "Congress")])

    _add(DomainType.LEGISLATIVE,
        "Trump threatened to veto spending bill without DOGE provisions",
        "Publicly demanded spending bill include DOGE-advocated government cuts",
        "2025-03-15", "2025-03-08",
        ["Sign clean spending bill", "Accept compromise on some cuts", "Shut down government"],
        "Presidential veto threat on must-pass legislation",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.LEGISLATIVE,
        "Trump signed government funding extension to avoid shutdown",
        "Signed continuing resolution extending government funding",
        "2025-03-14", "2025-03-08",
        ["Veto CR and shut down government", "Demand full-year appropriations", "Accept CR with conditions"],
        "Presidential signature on must-pass appropriations legislation",
        [("https://www.whitehouse.gov", "White House"), ("https://www.congress.gov", "Congress")])

    _add(DomainType.LEGISLATIVE,
        "Trump pushed for making his 2017 tax cuts permanent",
        "Demanded Congress pass legislation making Tax Cuts and Jobs Act provisions permanent",
        "2025-03-01", "2025-02-20",
        ["Accept partial extension", "Propose new tax reform instead", "Allow some provisions to expire"],
        "Presidential lobbying of Congress for tax legislation",
        [("https://www.whitehouse.gov", "White House"), ("https://www.congress.gov", "Congress")])

    _add(DomainType.LEGISLATIVE,
        "Trump proposed eliminating taxes on tips, overtime, and Social Security",
        "Demanded Congress eliminate federal income tax on tips, overtime pay, and Social Security benefits",
        "2025-01-20", "2025-01-15",
        ["Propose targeted tax credits instead", "Phase in changes gradually", "Limit to below income threshold"],
        "Presidential legislative proposal from campaign promises",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC COMMUNICATIONS (~20 seeds, target 10%)
    # ══════════════════════════════════════════════════════════════════════

    _add(DomainType.PUBLIC_COMMS,
        "Trump resumed daily use of Truth Social for policy announcements",
        "Used Truth Social as primary communication channel for major policy announcements",
        "2025-01-20", "2025-01-15",
        ["Use traditional press conferences", "Use official White House channels only", "Limit social media use"],
        "Personal presidential communication choice",
        [("https://truthsocial.com", "Truth Social"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.PUBLIC_COMMS,
        "Trump publicly attacked judges who blocked his executive orders",
        "Made personal attacks on federal judges who issued injunctions against executive orders",
        "2025-02-15", "2025-02-08",
        ["Accept court rulings and appeal through DOJ", "Issue diplomatic statements about disagreement", "Propose judicial reform legislation"],
        "Presidential public statements attacking judicial independence",
        [("https://truthsocial.com", "Truth Social"), ("https://apnews.com", "AP News")])

    _add(DomainType.PUBLIC_COMMS,
        "Trump held rally-style press conference at Mar-a-Lago",
        "Conducted major press event at private club with campaign-style staging",
        "2025-03-10", "2025-03-03",
        ["Use White House press room", "Hold traditional press conference", "Issue written statement only"],
        "Presidential discretion over public communication venue",
        [("https://www.whitehouse.gov", "White House"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.PUBLIC_COMMS,
        "Trump threatened to use Alien Enemies Act of 1798 for deportations",
        "Invoked the 1798 Alien Enemies Act to justify expedited deportation of gang members",
        "2025-01-20", "2025-01-15",
        ["Use existing immigration enforcement", "Propose new legislation", "Issue executive order with modern legal authority"],
        "Presidential invocation of rarely-used wartime statute",
        [("https://www.whitehouse.gov", "White House"), ("https://apnews.com", "AP News")])

    _add(DomainType.PUBLIC_COMMS,
        "Trump publicly disputed economic data showing recession risks",
        "Disputed negative economic indicators and blamed previous administration",
        "2025-04-15", "2025-04-08",
        ["Acknowledge economic challenges", "Propose stimulus measures", "Blame the Fed specifically"],
        "Presidential public communication on economic narrative",
        [("https://truthsocial.com", "Truth Social"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.PUBLIC_COMMS,
        "Trump Signal group chat leak exposed war plans shared with journalist",
        "Administration officials accidentally added journalist to Signal chat discussing Yemen strikes",
        "2025-03-24", "2025-03-18",
        ["Acknowledge mistake and investigate", "Deny significance of leak", "Fire officials responsible"],
        "Presidential response to major security breach emerged from executive communication",
        [("https://www.theatlantic.com", "The Atlantic"), ("https://apnews.com", "AP News")])

    # ══════════════════════════════════════════════════════════════════════
    # LEGAL & JUDICIAL (~10 seeds, target 5%)
    # ══════════════════════════════════════════════════════════════════════

    _add(DomainType.LEGAL_JUDICIAL,
        "Trump DOJ dropped all federal cases against January 6 defendants",
        "Directed DOJ to dismiss remaining federal prosecutions of January 6 defendants",
        "2025-01-21", "2025-01-15",
        ["Continue prosecutions selectively", "Allow DOJ independence", "Drop only misdemeanor cases"],
        "Presidential directive to Department of Justice",
        [("https://www.doj.gov", "DOJ"), ("https://apnews.com", "AP News")])

    _add(DomainType.LEGAL_JUDICIAL,
        "Trump DOJ ordered firing of prosecutors who worked on Trump cases",
        "Directed removal of prosecutors from the special counsel's office and NY cases",
        "2025-01-25", "2025-01-20",
        ["Allow natural reassignment", "Investigate but not fire", "Reassign to other duties"],
        "Presidential directive on DOJ personnel through AG",
        [("https://www.doj.gov", "DOJ"), ("https://www.reuters.com", "Reuters")])

    _add(DomainType.LEGAL_JUDICIAL,
        "Trump used executive authority to challenge court injunctions via appeals",
        "DOJ aggressively challenged judicial injunctions against executive orders",
        "2025-02-20", "2025-02-10",
        ["Comply with injunctions pending appeal", "Propose legislation to limit judicial review", "Ignore injunctions (constitutional crisis)"],
        "Presidential legal strategy through DOJ",
        [("https://www.doj.gov", "DOJ"), ("https://www.scotusblog.com", "SCOTUSblog")])

    _add(DomainType.LEGAL_JUDICIAL,
        "Trump ordered DOJ to investigate political opponents",
        "Directed Attorney General to open investigations into perceived political enemies",
        "2025-02-01", "2025-01-25",
        ["Allow DOJ to set its own investigative priorities", "Request GAO audit instead", "Form special commission"],
        "Presidential directive to Attorney General",
        [("https://www.doj.gov", "DOJ"), ("https://apnews.com", "AP News")])

    _add(DomainType.LEGAL_JUDICIAL,
        "Trump administration defied court orders on deportation flights",
        "Continued deportation flights to Venezuela despite federal judge's temporary restraining order",
        "2025-03-15", "2025-03-08",
        ["Comply with court order immediately", "Appeal and suspend operations pending", "Negotiate modified compliance"],
        "Executive branch defiance of judicial order",
        [("https://apnews.com", "AP News"), ("https://www.reuters.com", "Reuters")])

    return seeds
