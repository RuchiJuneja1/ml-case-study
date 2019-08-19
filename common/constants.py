dropped_columns_due_to_missing_data = ['account_days_in_dc_12_24m',
									   'account_days_in_rem_12_24m',
									   'account_days_in_term_12_24m',
									   'account_incoming_debt_vs_paid_0_24m',
									   'account_status',
									   'account_worst_status_0_3m',
									   'account_worst_status_12_24m',
									   'account_worst_status_3_6m',
									   'account_worst_status_6_12m',
									   'avg_payment_span_0_12m',
									   'avg_payment_span_0_3m',
									   'num_active_div_by_paid_inv_0_12m',
									   'num_arch_written_off_0_12m',
									   'num_arch_written_off_12_24m',
									   'worst_status_active_inv']

aggregation_dict = {
	'num_arch_ok': {'num_arch_ok_0_12m': 1.0, 'num_arch_ok_12_24m': 1.0},
	'num_arch_dc': {'num_arch_dc_0_12m': 1.0, 'num_arch_dc_12_24m': 1.0},
	'max_paid_inv': {'max_paid_inv_0_12m': 1.0, 'max_paid_inv_0_24m': 1.0},
}

correlated_drop_feature_list = ['sum_capital_paid_account_0_12m',
								'sum_capital_paid_account_12_24m',
								'num_active_inv',
								'num_arch_rem_0_12m',
								'sum_paid_inv_0_12m',
								'name_in_email',
								'has_paid']

categorical_parameters = ['account_status',
						  'account_worst_status_0_3m',
						  'account_worst_status_12_24m',
						  'account_worst_status_3_6m',
						  'account_worst_status_6_12m',
						  'merchant_category',
						  'merchant_group',
						  'name_in_email',
						  'status_last_archived_0_24m',
						  'status_2nd_last_archived_0_24m',
						  'status_3rd_last_archived_0_24m',
						  'status_max_archived_0_6_months',
						  'status_max_archived_0_12_months',
						  'status_max_archived_0_24_months',
						  'worst_status_active_inv',
						  'has_paid']

numerical_parameters = ['account_amount_added_12_24m',
						'account_days_in_dc_12_24m',
						'account_days_in_rem_12_24m',
						'account_days_in_term_12_24m',
						'account_incoming_debt_vs_paid_0_24m',
						'age',
						'avg_payment_span_0_12m',
						'avg_payment_span_0_3m',
						'max_paid_inv_0_12m',
						'max_paid_inv_0_24m',
						'num_active_div_by_paid_inv_0_12m',
						'num_active_inv',
						'num_arch_dc_0_12m',
						'num_arch_dc_12_24m',
						'num_arch_ok_0_12m',
						'num_arch_ok_12_24m',
						'num_arch_rem_0_12m',
						'num_arch_written_off_0_12m',
						'num_arch_written_off_12_24m',
						'num_unpaid_bills',
						'recovery_debt',
						'sum_capital_paid_account_0_12m',
						'sum_capital_paid_account_12_24m',
						'sum_paid_inv_0_12m',
						'time_hours']

pmi_drop_column_list = ['status_max_archived_0_6_months_3',
						'status_3rd_last_archived_0_24m_3',
						'merchant_category_Youthful Shoes & Clothing',
						'merchant_category_Dating services',
						'merchant_group_Erotic Materials',
						'merchant_group_Intangible products',
						'merchant_category_Body & Hair Care',
						'merchant_category_Prints & Photos',
						'merchant_category_Food & Beverage',
						'merchant_category_Children toys',
						'merchant_category_Music & Movies',
						'status_max_archived_0_6_months_2',
						'merchant_category_Video Games & Related accessories',
						'merchant_group_Health & Beauty',
						'merchant_category_Fragrances',
						'merchant_category_Hobby articles',
						'merchant_category_Diversified erotic material',
						'merchant_category_Cleaning & Sanitary',
						'merchant_group_Jewelry & Accessories',
						'status_2nd_last_archived_0_24m_2',
						'merchant_group_Food & Beverage',
						'merchant_category_Costumes & Party supplies',
						'merchant_category_Diversified Jewelry & Accessories',
						'merchant_category_Books & Magazines',
						'merchant_category_Pharmaceutical products',
						'merchant_group_Electronics',
						'merchant_category_Sex toys',
						'status_last_archived_0_24m_2',
						'merchant_category_Wine, Beer & Liquor',
						'merchant_category_Prescription optics',
						'status_2nd_last_archived_0_24m_5',
						'merchant_group_Leisure, Sport & Hobby',
						'merchant_category_Children Clothes & Nurturing products',
						'merchant_category_Decoration & Art',
						'merchant_category_Collectibles',
						'status_max_archived_0_24_months_5',
						'merchant_category_Kitchenware',
						'status_3rd_last_archived_0_24m_2',
						'merchant_category_Sports gear & Outdoor',
						'merchant_category_Diversified Health & Beauty products',
						'status_3rd_last_archived_0_24m_5',
						'merchant_group_Home & Garden',
						'status_last_archived_0_24m_5',
						'status_max_archived_0_12_months_5',
						'merchant_category_Automotive Parts & Accessories',
						'merchant_group_Children Products',
						'merchant_category_Wheels & Tires',
						'merchant_category_Car electronics',
						'merchant_category_Concept stores & Miscellaneous',
						'merchant_category_Cosmetics',
						'merchant_category_Dietary supplements',
						'merchant_category_Digital services',
						'merchant_category_Diversified Home & Garden products',
						'merchant_category_Diversified children products',
						'merchant_category_Diversified electronics',
						'merchant_category_Education',
						'merchant_category_Electronic equipment & Related accessories',
						'merchant_category_Erotic Clothing & Accessories',
						'merchant_category_Event tickets',
						'merchant_category_Furniture',
						'merchant_category_General Shoes & Clothing',
						'merchant_category_Household electronics (whitegoods/appliances)',
						'merchant_category_Jewelry & Watches',
						'merchant_category_Musical Instruments & Equipment',
						'merchant_category_Non',
						'merchant_category_Office machines & Related accessories (excl. computers)',
						'merchant_category_Personal care & Body improvement',
						'merchant_category_Pet supplies',
						'merchant_category_Plants & Flowers',
						'merchant_category_Bags & Wallets',
						'merchant_category_Tobacco',
						'merchant_category_Tools & Home improvement',
						'merchant_category_Travel services',
						'merchant_category_Underwear',
						'merchant_category_Safety products']

non_important_features = ['status_max_archived_0_12_months_3',
						  'status_max_archived_0_12_months_1',
						  'merchant_category_Diversified entertainment',
						  'status_max_archived_0_24_months_2',
						  'status_max_archived_0_24_months_3',
						  'status_max_archived_0_24_months_1',
						  'status_2nd_last_archived_0_24m_1',
						  'merchant_group_Clothing & Shoes']

model_column_list = ['num_arch_ok',
					 'max_paid_inv',
					 'num_unpaid_bills',
					 'time_hours',
					 'age',
					 'num_arch_dc',
					 'account_amount_added_12_24m',
					 'status_last_archived_0_24m_3',
					 'merchant_group_Entertainment',
					 'recovery_debt',
					 'status_3rd_last_archived_0_24m_1',
					 'status_last_archived_0_24m_1',
					 'status_2nd_last_archived_0_24m_3',
					 'status_max_archived_0_12_months_2',
					 'status_max_archived_0_6_months_1']

required_input_file_parameter = ['has_paid',
								'default',
								'account_status',
								'account_worst_status_0_3m',
								'account_worst_status_12_24m',
								'account_worst_status_3_6m',
								'account_worst_status_6_12m',
								'merchant_category',
								'merchant_group',
								'name_in_email',
								'status_last_archived_0_24m',
								'status_2nd_last_archived_0_24m',
								'status_3rd_last_archived_0_24m',
								'status_max_archived_0_6_months',
								'status_max_archived_0_12_months',
								'status_max_archived_0_24_months',
								'worst_status_active_inv',
								'account_amount_added_12_24m',
								'account_days_in_dc_12_24m',
								'account_days_in_rem_12_24m',
								'account_days_in_term_12_24m',
								'account_incoming_debt_vs_paid_0_24m',
								'age',
								'avg_payment_span_0_12m',
								'avg_payment_span_0_3m',
								'max_paid_inv_0_12m',
								'max_paid_inv_0_24m',
								'num_active_div_by_paid_inv_0_12m',
								'num_active_inv',
								'num_arch_dc_0_12m',
								'num_arch_dc_12_24m',
								'num_arch_ok_0_12m',
								'num_arch_ok_12_24m',
								'num_arch_rem_0_12m',
								'num_arch_written_off_0_12m',
								'num_arch_written_off_12_24m',
								'num_unpaid_bills',
								'recovery_debt',
								'sum_capital_paid_account_0_12m',
								'sum_capital_paid_account_12_24m',
								'sum_paid_inv_0_12m',
								'time_hours',
								'uuid']
